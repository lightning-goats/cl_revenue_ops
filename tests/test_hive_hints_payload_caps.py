"""P1-011: consumer-side byte-size + entry-count caps on the untrusted
["hive","hints"] datastore value.

The datastore value is locally writable and producer-controlled; an oversized
value must be rejected before json.loads so the fee path degrades to neutral
instead of hanging on an adversarial payload.

Entry-count is different: cl-hive legitimately emits one hint per
gossip-known peer (hundreds on a well-connected node), so an over-cap hint
map is no longer rejected wholesale (that silently killed the entire hint
pipeline in production — see 2026-07 P0). Instead the per-peer map is
truncated to MAX_PEERS_IN_SNAPSHOT entries, prioritizing fleet members and
peers with actionable content (channel_open_hint, closure_recommended,
corridor_role) over everything else.
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

    def _mixed_snap(self, n_members, n_actionable, n_filler):
        """n_members "member" peers, n_actionable non-member peers with an
        actionable channel_open_hint, and n_filler plain/neutral peers.
        Keys are zero-padded hex so sorted() order is deterministic and
        distinct across the three groups (members < actionable < filler
        lexically has no bearing on priority — priority is tier-based, not
        key-based; the zero-padding just keeps ids unique and orderable).
        """
        hints = {}
        for i in range(n_members):
            hints[f"m{i:065x}"] = {"member": True}
        for i in range(n_actionable):
            hints[f"a{i:065x}"] = {
                "member": False,
                "channel_open_hint": {"open_preference": "open"},
            }
        for i in range(n_filler):
            hints[f"f{i:065x}"] = {"member": False}
        return {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": hints,
        }

    def test_over_entry_cap_truncated_not_rejected(self, mock_plugin):
        # Behavior change (2026-07 P0 fix): an over-cap hint map used to be
        # rejected wholesale, which silently killed the entire hint pipeline
        # in production once cl-hive started emitting one hint per
        # gossip-known peer (hundreds, not <=200). It must now be truncated
        # and the snapshot kept usable.
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT + 1)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True
        assert len(adapter._snapshot["hints"]) == HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT

    def test_at_entry_cap_accepted(self, mock_plugin):
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True
        assert len(adapter._snapshot["hints"]) == HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT

    def test_under_entry_cap_unchanged(self, mock_plugin):
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT - 1)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True
        assert len(adapter._snapshot["hints"]) == HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT - 1

    def test_members_and_actionable_survive_truncation(self, mock_plugin):
        # 5 members + 10 actionable + 235 filler = 250 total, cap 200.
        snap = self._mixed_snap(n_members=5, n_actionable=10, n_filler=235)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_usable() is True
        surviving = adapter._snapshot["hints"]
        assert len(surviving) == HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT
        member_keys = {k for k in snap["hints"] if k.startswith("m")}
        actionable_keys = {k for k in snap["hints"] if k.startswith("a")}
        assert member_keys <= surviving.keys()
        assert actionable_keys <= surviving.keys()

    def test_truncated_snapshot_membership_getter_works(self, mock_plugin):
        snap = self._mixed_snap(n_members=5, n_actionable=10, n_filler=235)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        assert adapter.is_hive_member("m" + "0" * 65) is True
        assert adapter.is_hive_member("a" + "0" * 65) is False

    def test_truncated_snapshot_open_candidates_works(self, mock_plugin):
        snap = self._mixed_snap(n_members=5, n_actionable=10, n_filler=235)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        candidates = dict(adapter.get_open_candidates())
        actionable_keys = {k for k in snap["hints"] if k.startswith("a")}
        assert set(candidates.keys()) == actionable_keys

    def test_truncation_is_deterministic(self, mock_plugin):
        snap = self._mixed_snap(n_members=5, n_actionable=10, n_filler=235)

        adapter_a = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter_a.poll()
        adapter_b = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter_b.poll()

        assert set(adapter_a._snapshot["hints"].keys()) == set(
            adapter_b._snapshot["hints"].keys()
        )

    def test_truncation_logs_one_warn(self, mock_plugin):
        snap = self._snap_with_n_hints(HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT + 50)
        adapter = _adapter(mock_plugin, {"string": json.dumps(snap)})
        adapter.poll()
        warn_calls = [
            call
            for call in mock_plugin.log.call_args_list
            if call.kwargs.get("level") == "warn"
            or (len(call.args) > 1 and call.args[1] == "warn")
        ]
        assert len(warn_calls) == 1
        msg = warn_calls[0].args[0]
        assert "truncated" in msg
        assert f"{HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT + 50}->{HiveHintAdapter.MAX_PEERS_IN_SNAPSHOT}" in msg
