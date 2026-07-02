"""P1-011: consumer-side byte-size + entry-count caps on the untrusted
["hive","hints"] datastore value.

The datastore value is locally writable and producer-controlled; an oversized
value must be rejected before json.loads/_normalize_hint_map so the fee path
degrades to neutral instead of hanging on an adversarial payload.
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
    # No live-export fallback available.
    plugin.rpc.call.return_value = None
    return adapter


class TestByteSizeCap:
    def test_multi_mb_payload_rejected_before_parse(self, mock_plugin):
        huge = "{" + ("x" * (HiveHintAdapter.DATASTORE_MAX_BYTES + 5_000_000))
        adapter = _adapter(mock_plugin, {"string": huge})
        adapter.poll()
        assert adapter.is_usable() is False
        # Fee path degrades to neutral (1.0), not an exception/hang.
        assert adapter.get_fee_bias("02aabb") == 1.0

    def test_oversized_hex_payload_rejected(self, mock_plugin):
        huge_hex = "61" * (HiveHintAdapter.DATASTORE_MAX_BYTES + 1000)
        adapter = _adapter(mock_plugin, {"hex": huge_hex})
        adapter.poll()
        assert adapter.is_usable() is False
        assert adapter.get_fee_bias("02aabb") == 1.0

    def test_payload_under_cap_still_parsed(self, mock_plugin):
        snap = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {"02aabb": {"member": True, "corridor_role": "owner",
                                 "competition_bias": 1, "traffic_confidence": 0.9}},
        }
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True


class TestEntryCountCap:
    def _snap_with_n_hints(self, n):
        return {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {f"{i:066x}": {"member": True} for i in range(n)},
        }

    def test_over_entry_cap_rejected(self, mock_plugin):
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT + 1)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is False
        assert adapter.get_fee_bias("00" * 33) == 1.0

    def test_over_entry_cap_not_normalized(self, mock_plugin):
        # The len() guard must short-circuit before _normalize_hint_map scans
        # every entry (no O(n) walk of an adversarial map).
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT + 50)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        called = {"n": 0}
        orig = HiveHintAdapter._normalize_hint_map

        def spy(hints):
            called["n"] += 1
            return orig(hints)

        HiveHintAdapter._normalize_hint_map = staticmethod(spy)
        try:
            adapter.poll()
        finally:
            HiveHintAdapter._normalize_hint_map = staticmethod(orig)
        assert called["n"] == 0

    def test_at_entry_cap_accepted(self, mock_plugin):
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True
