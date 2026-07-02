"""P1-022: metabolic/immune section TTL clamped to the snapshot TTL upper
bound (HINT_MAX_TTL_SECONDS) so a huge section ttl_seconds cannot defeat
section-freshness.
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


def _adapter(plugin, datastore_entry):
    adapter = HiveHintAdapter(plugin)
    adapter.data_service = MagicMock()
    adapter.data_service.list_datastore.return_value = {"datastore": [datastore_entry]}
    plugin.rpc.call.return_value = None
    return adapter


class TestMetabolicTtlClamp:
    def test_ttl_clamped_to_snapshot_upper_bound(self):
        clamped = HiveHintAdapter._metabolic_ttl_for({"ttl_seconds": 31_536_000}, None)
        assert clamped == HiveHintAdapter.HINT_MAX_TTL_SECONDS

    def test_snapshot_ttl_also_clamped(self):
        clamped = HiveHintAdapter._metabolic_ttl_for({}, {"ttl_seconds": 31_536_000})
        assert clamped == HiveHintAdapter.HINT_MAX_TTL_SECONDS

    def test_old_section_with_huge_ttl_is_stale(self, mock_plugin):
        now = int(time.time())
        snap = {
            "generated_at": now,          # outer snapshot is fresh
            "ttl_seconds": 900,
            "hints": {"02aabb": {"member": True}},
            "metabolic_influence": {
                "schema_version": "metabolic_influence/v1",
                "generated_at": now - 100_000,   # ~27h old
                "ttl_seconds": 31_536_000,        # 1 year -> would defeat freshness
                "confidence": "high",
                "peer_effects": {},
            },
        }
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        status = adapter.get_metabolic_status()
        assert status["fresh"] is False
