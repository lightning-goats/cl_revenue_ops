"""P1-012: type-confusion on raw operator params returns a clean error dict.

re.match()/int() run on raw operator params must not leak a
TypeError/ValueError traceback when the param is the wrong type.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


def test_analyze_non_str_channel_id(mod):
    mod.flow_analyzer = MagicMock()
    res = mod.revenue_analyze(mod.plugin, channel_id=12345)
    assert isinstance(res, dict) and "error" in res


def test_set_fee_non_str_channel_id(mod):
    mod.fee_controller = MagicMock()
    mod.config = MagicMock(min_fee_ppm=1, max_fee_ppm=1000)
    res = mod.revenue_set_fee(mod.plugin, channel_id=999, fee_ppm=100)
    assert isinstance(res, dict) and res.get("status") == "error"


def test_rebalance_non_str_channel(mod):
    mod.rebalancer = MagicMock()
    res = mod.revenue_rebalance(mod.plugin, from_channel=1, to_channel=2, amount_sats=1000)
    assert isinstance(res, dict) and res.get("status") == "error"


def test_dashboard_non_int_window(mod):
    mod.profitability_analyzer = MagicMock()
    mod.database = MagicMock()
    res = mod.revenue_dashboard(mod.plugin, window_days="x")
    assert isinstance(res, dict) and "error" in res
