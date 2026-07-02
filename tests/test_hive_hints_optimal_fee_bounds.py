"""P1-023: get_optimal_fee_estimate bounded to [1, MAX_FLEET_FEE_PRIOR_PPM]
like get_fleet_fee_prior; an out-of-range estimate neutralizes to 0.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


def _adapter_with_estimate(plugin, value):
    snap = {
        "generated_at": int(time.time()),
        "ttl_seconds": 900,
        "hints": {"02aabb": {"member": True, "optimal_fee_estimate_ppm": value}},
    }
    adapter = HiveHintAdapter(plugin)
    adapter.data_service = MagicMock()
    adapter.data_service.list_datastore.return_value = {"datastore": [{"string": json.dumps(snap)}]}
    plugin.rpc.call.return_value = None
    adapter.poll()
    return adapter


class TestOptimalFeeEstimateBounds:
    def test_out_of_range_estimate_rejected(self, mock_plugin):
        adapter = _adapter_with_estimate(mock_plugin, 50_000)
        assert adapter.get_optimal_fee_estimate("02aabb") == 0

    def test_in_range_estimate_returned(self, mock_plugin):
        adapter = _adapter_with_estimate(mock_plugin, 500)
        assert adapter.get_optimal_fee_estimate("02aabb") == 500

    def test_upper_bound_estimate_returned(self, mock_plugin):
        adapter = _adapter_with_estimate(mock_plugin, HiveHintAdapter.MAX_FLEET_FEE_PRIOR_PPM)
        assert adapter.get_optimal_fee_estimate("02aabb") == HiveHintAdapter.MAX_FLEET_FEE_PRIOR_PPM
