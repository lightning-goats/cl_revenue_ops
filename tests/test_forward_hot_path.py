"""Tests for per-forward hot-path optimizations.

- Settled forwards coalesce the forward insert + reputation upsert into one
  transaction via database.record_forward_and_reputation when available,
  falling back to the legacy two autocommit writes when absent.
- _resolve_scid_to_peer negatively caches unknown SCIDs so a burst of
  forwards on a foreign/closed channel doesn't rebuild the SCID map per event.
- Startup hydration prefers a paged listforwards(index="created") loop and
  falls back to the legacy full settled fetch on any error.
"""

from unittest.mock import MagicMock

import pytest

from modules.fee_authority import FeeAuthorityGate
from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "c" * 64
SCID = "100x1x0"


def _settled_event():
    return {
        "status": "settled",
        "in_channel": SCID,
        "out_channel": "200x2x0",
        "in_msat": 1_000_000,
        "out_msat": 999_000,
        "fee_msat": 1_000,
        "received_time": 1_700_000_000,
        "resolved_time": 1_700_000_002,
    }


def _module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = None
    return mod


def _disable_fee_authority(mod):
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 10_000)
    mod.fee_authority_gate.set_enabled(False, reason="setconfig")


class TestForwardWriteCoalescing:
    def test_settled_forward_uses_single_combined_write(self):
        mod = _module()
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.record_forward_and_reputation.assert_called_once()
        forward, peer_id, success = db.record_forward_and_reputation.call_args.args
        assert peer_id == PEER
        assert success is True
        assert forward["in_channel"] == SCID
        assert forward["out_channel"] == "200x2x0"
        assert forward["in_msat"] == 1_000_000
        assert forward["out_msat"] == 999_000
        assert forward["fee_msat"] == 1_000
        assert forward["received_time"] == 1_700_000_000
        assert forward["resolved_time"] == 1_700_000_002
        assert forward["resolution_time"] == 2
        # Legacy two-write path must not run.
        db.record_forward.assert_not_called()
        db.update_peer_reputation.assert_not_called()

    def test_settled_forward_falls_back_to_two_writes_when_method_absent(self):
        mod = _module()
        db = MagicMock(spec=["record_forward", "update_peer_reputation"])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.update_peer_reputation.assert_called_once_with(PEER, is_success=True)
        db.record_forward.assert_called_once_with(
            SCID, "200x2x0", 1_000_000, 999_000, 1_000,
            1_700_000_000, 1_700_000_002, 2,
        )

    def test_settled_forward_without_peer_records_forward_only(self):
        mod = _module()
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=None)

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.record_forward_and_reputation.assert_not_called()
        db.update_peer_reputation.assert_not_called()
        db.record_forward.assert_called_once()

    def test_settled_forward_wakes_governed_fee_loop_after_persist(self):
        mod = _module()
        order = []
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward",
            "update_peer_reputation",
        ])
        db.record_forward_and_reputation.side_effect = (
            lambda *args: order.append("persist")
        )
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)
        mod.fee_controller = MagicMock()
        mod.fee_controller.should_wake_acquisition_cycle.side_effect = (
            lambda *args: order.append("evaluate") or True
        )
        mod._request_fee_adjustment_wake = lambda: order.append("wake")

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        assert order == ["persist", "evaluate", "wake"]
        mod.fee_controller.should_wake_acquisition_cycle.assert_called_once_with(
            "200x2x0", 999_000
        )

    def test_settled_forward_monitor_failure_is_neutral(self):
        mod = _module()
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward",
            "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)
        mod.fee_controller = MagicMock()
        mod.fee_controller.should_wake_acquisition_cycle.side_effect = (
            ValueError("malformed monitor state")
        )
        mod._request_fee_adjustment_wake = MagicMock()

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.record_forward_and_reputation.assert_called_once()
        mod._request_fee_adjustment_wake.assert_not_called()

    def test_settled_forward_can_wake_yield_inventory_after_persist(self):
        mod = _module()
        order = []
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward",
            "update_peer_reputation",
        ])
        db.record_forward_and_reputation.side_effect = (
            lambda *args: order.append("persist")
        )
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)
        mod.fee_controller = MagicMock()
        mod.fee_controller.should_wake_acquisition_cycle.side_effect = (
            lambda *args: order.append("acquisition") or False
        )
        mod.fee_controller.should_wake_yield_inventory_cycle.side_effect = (
            lambda *args: order.append("yield") or True
        )
        mod._request_fee_adjustment_wake = lambda: order.append("wake")

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        assert order == ["persist", "acquisition", "yield", "wake"]
        mod.fee_controller.should_wake_yield_inventory_cycle.assert_called_once_with(
            "200x2x0", 999_000
        )

    def test_disabled_authority_does_not_wake_acquisition_monitor(self):
        mod = _module()
        _disable_fee_authority(mod)
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward",
            "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)
        mod.fee_controller = MagicMock()
        mod._request_fee_adjustment_wake = MagicMock()

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.record_forward_and_reputation.assert_called_once()
        mod.fee_controller.should_wake_acquisition_cycle.assert_not_called()
        mod.fee_controller.should_wake_yield_inventory_cycle.assert_not_called()
        mod._request_fee_adjustment_wake.assert_not_called()

    def test_failed_forward_still_penalizes_reputation_immediately(self):
        mod = _module()
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)

        event = _settled_event()
        event["status"] = "failed"
        mod._on_forward_event_impl(event, mod.plugin)

        db.update_peer_reputation.assert_called_once_with(PEER, is_success=False)
        db.record_forward_and_reputation.assert_not_called()
        db.record_forward.assert_not_called()

    @pytest.mark.parametrize("status", ["failed", "local_failed"])
    def test_disabled_authority_blocks_failed_forward_fee_trigger(self, status):
        mod = _module()
        _disable_fee_authority(mod)
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)
        mod.fee_controller = MagicMock()
        mod.fee_controller.is_fee_relevant_failure.return_value = True
        mod.fee_controller._state_lock.acquire.return_value = True
        mod.fee_controller._channel_fee_states = {
            "200x2x0": MagicMock(last_fee_ppm=250),
        }

        event = _settled_event()
        event.update({
            "status": status,
            "failcode": 0x1000 | 12,
            "failreason": "WIRE_FEE_INSUFFICIENT",
        })
        mod._on_forward_event_impl(event, mod.plugin)

        mod.fee_controller.is_fee_relevant_failure.assert_not_called()
        mod.fee_controller._state_lock.acquire.assert_not_called()
        mod.fee_controller.record_failed_forward.assert_not_called()
        if status == "failed":
            db.update_peer_reputation.assert_called_once_with(
                PEER, is_success=False
            )

    def test_disabled_authority_keeps_settled_forward_accounting_live(self):
        mod = _module()
        _disable_fee_authority(mod)
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=PEER)

        mod._on_forward_event_impl(_settled_event(), mod.plugin)

        db.record_forward_and_reputation.assert_called_once()
        forward, peer_id, success = db.record_forward_and_reputation.call_args.args
        assert (forward["out_channel"], peer_id, success) == (
            "200x2x0", PEER, True
        )


class TestScidNegativeCache:
    def _module_with_channels(self, channels):
        mod = _module()
        ds = MagicMock()
        ds.get_peer_channels.return_value = {"channels": channels}
        mod.data_service = ds
        return mod

    def test_unknown_scid_is_negatively_cached(self):
        mod = self._module_with_channels([])

        assert mod._resolve_scid_to_peer("999x9x9") is None
        assert mod._resolve_scid_to_peer("999x9x9") is None
        assert mod._resolve_scid_to_peer("999x9x9") is None

        # Only the first miss rebuilds the SCID map.
        assert mod.data_service.get_peer_channels.call_count == 1

    def test_known_scid_resolves_and_is_cached(self):
        mod = self._module_with_channels([
            {"short_channel_id": SCID, "peer_id": PEER},
        ])

        assert mod._resolve_scid_to_peer(SCID) == PEER
        assert mod._resolve_scid_to_peer(SCID) == PEER
        assert mod.data_service.get_peer_channels.call_count == 1

    def test_negative_cache_is_bounded(self):
        mod = self._module_with_channels([])
        mod._SCID_NEGATIVE_CACHE_MAX_ENTRIES = 2

        assert mod._resolve_scid_to_peer("1x1x1") is None
        assert mod._resolve_scid_to_peer("2x2x2") is None
        assert mod._resolve_scid_to_peer("3x3x3") is None

        negatives = [k for k, v in mod._scid_to_peer_cache.items() if v is None]
        assert len(negatives) == 2

    def test_hourly_clear_flushes_negative_entries(self):
        mod = self._module_with_channels([])

        assert mod._resolve_scid_to_peer("999x9x9") is None
        assert mod.data_service.get_peer_channels.call_count == 1

        # Simulate the hourly TTL elapsing; the negative entry must be retried.
        mod._scid_cache_last_cleared = 0
        assert mod._resolve_scid_to_peer("999x9x9") is None
        assert mod.data_service.get_peer_channels.call_count == 2


class TestHydrationPagedFetch:
    def test_paged_path_consumed_when_rpc_supports_it(self):
        mod = _module()
        mod._HYDRATION_PAGE_LIMIT = 2
        mod.data_service = MagicMock()

        pages = {
            0: {"forwards": [
                {"created_index": 1, "received_time": 100, "in_channel": "1x1x1"},
                {"created_index": 2, "received_time": 200, "in_channel": "2x2x2"},
            ]},
            3: {"forwards": [
                {"created_index": 3, "received_time": 300, "in_channel": "3x3x3"},
            ]},
        }

        def _listforwards(**kwargs):
            assert kwargs["status"] == "settled"
            assert kwargs["index"] == "created"
            assert kwargs["limit"] == 2
            return pages[kwargs["start"]]

        mod.safe_plugin = MagicMock()
        mod.safe_plugin.rpc.listforwards.side_effect = _listforwards

        result = mod._hydration_fetch_settled_forwards(start_time=150)

        assert [f["in_channel"] for f in result] == ["2x2x2", "3x3x3"]
        assert mod.safe_plugin.rpc.listforwards.call_count == 2
        # Full-history fallback fetch must not run.
        mod.data_service.get_forwards.assert_not_called()

    def test_fallback_to_full_fetch_on_any_error(self):
        mod = _module()
        mod.safe_plugin = MagicMock()
        # Older CLN: paged kwargs rejected.
        mod.safe_plugin.rpc.listforwards.side_effect = TypeError("unexpected keyword")
        mod.data_service = MagicMock()
        mod.data_service.get_forwards.return_value = {
            "forwards": [
                {"received_time": 100, "in_channel": "1x1x1"},
                {"received_time": 200, "in_channel": "2x2x2"},
            ]
        }

        result = mod._hydration_fetch_settled_forwards(start_time=150)

        assert [f["in_channel"] for f in result] == ["2x2x2"]
        mod.data_service.get_forwards.assert_called_once_with(status="settled")

    def test_fallback_when_paging_does_not_advance(self):
        mod = _module()
        mod._HYDRATION_PAGE_LIMIT = 1
        mod.safe_plugin = MagicMock()
        # Pathological mock: same full page (same created_index) forever.
        mod.safe_plugin.rpc.listforwards.return_value = {
            "forwards": [{"created_index": 0, "received_time": 200, "in_channel": "1x1x1"}]
        }
        mod.data_service = MagicMock()
        mod.data_service.get_forwards.return_value = {"forwards": []}

        result = mod._hydration_fetch_settled_forwards(start_time=100)

        assert result == []
        mod.data_service.get_forwards.assert_called_once_with(status="settled")

    def test_both_paths_failing_returns_empty(self):
        mod = _module()
        mod.safe_plugin = MagicMock()
        mod.safe_plugin.rpc.listforwards.side_effect = RuntimeError("rpc down")
        mod.data_service = MagicMock()
        mod.data_service.get_forwards.side_effect = RuntimeError("rpc down")

        assert mod._hydration_fetch_settled_forwards(start_time=0) == []
