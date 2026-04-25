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

    def test_hive_exploration_multiplier_uses_fee_elasticity(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_centrality.return_value = 0.05
        adapter.get_corridor_role.return_value = "owner"
        adapter.get_fee_elasticity.return_value = 0.25
        fc.hive_hints = adapter

        multiplier = fc._get_hive_exploration_multiplier("02aabb")

        assert multiplier > 1.5


class TestMemberFeePolicy:
    def test_hive_member_uses_advisory_dynamic_fee(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02member")
        assert result is None
        assert fc._consume_hive_member_advisory("02member") is True
        assert fc._consume_hive_member_advisory("02member") is False

    def test_non_member_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = False
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02nonmember")
        assert result is None

    def test_no_adapter_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        result = fc._check_hive_member_fee("02peer")
        assert result is None

    def test_grace_period_keeps_membership_advisory_after_stale(self, mock_plugin, mock_config, mock_database):
        """Stale-but-usable hints keep dynamic pricing fleet-aware without forcing 0-PPM."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        assert fc._check_hive_member_fee("02peer") is None
        assert fc._consume_hive_member_advisory("02peer") is True

        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        adapter.is_usable.return_value = True
        assert fc._check_hive_member_fee("02peer") is None
        assert fc._consume_hive_member_advisory("02peer") is True

    def test_explicit_hive_unavailable_clears_sticky_membership(self, mock_plugin, mock_config, mock_database):
        """When cl-hive is disabled, dynamic fee control should reprice immediately."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        assert fc._check_hive_member_fee("02peer") is None
        assert fc._consume_hive_member_advisory("02peer") is True

        adapter.is_hive_member.return_value = False
        adapter.is_usable.return_value = False

        assert fc._check_hive_member_fee("02peer") is None
        assert "02peer" not in fc._hive_member_set_at
        assert fc._consume_hive_member_advisory("02peer") is False
        assert fc._consume_hive_member_release("02peer") is True
        assert fc._consume_hive_member_release("02peer") is False

    def test_grace_period_expires(self, mock_plugin, mock_config, mock_database):
        """After grace period expires, revert to DTS+PID (return None)."""
        import time as _time
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        fc._check_hive_member_fee("02peer")
        fc._hive_member_set_at["02peer"] = int(_time.time()) - 1801
        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        adapter.is_usable.return_value = True
        assert fc._check_hive_member_fee("02peer") is None
        assert fc._consume_hive_member_advisory("02peer") is False
