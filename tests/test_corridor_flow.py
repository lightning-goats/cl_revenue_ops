"""Z-3: mycelial corridor utilization instrumentation.

Covers:
- Forward classification (internal_transit / edge_in / edge_out / external)
  using the same membership source as the hive-member fee gate.
- corridor_flow_daily upsert arithmetic in modules/database.py.
- The `mycelial_corridor` section on the revenue-dashboard RPC.

This is a success/utilization metric only -- it must never gate, warn, or
revoke anything, so these tests only assert counters/shape, not any
threshold behavior.
"""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from modules.database import Database
from tests.plugin_test_utils import load_plugin_module


def _make_db():
    path = os.path.join(tempfile.mkdtemp(prefix="corridor_flow_test_"), "test.db")
    db = Database(path, MagicMock())
    db.initialize()
    return db


HIVE_A = "02" + "aa" * 32
HIVE_B = "02" + "bb" * 32
EXT_C = "02" + "cc" * 32
EXT_D = "02" + "dd" * 32


def _module_with_members(members):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    fc = MagicMock()

    def _status(peer_id):
        return {"member": peer_id in members}

    fc._get_hive_membership_status = MagicMock(side_effect=_status)
    mod.fee_controller = fc
    return mod


class TestCorridorClassification:
    def test_hive_to_hive_is_internal_transit(self):
        mod = _module_with_members({HIVE_A, HIVE_B})
        assert mod._corridor_classify_forward(HIVE_A, HIVE_B) == "internal_transit"

    def test_external_to_hive_is_edge_in(self):
        mod = _module_with_members({HIVE_B})
        assert mod._corridor_classify_forward(EXT_C, HIVE_B) == "edge_in"

    def test_hive_to_external_is_edge_out(self):
        mod = _module_with_members({HIVE_A})
        assert mod._corridor_classify_forward(HIVE_A, EXT_D) == "edge_out"

    def test_external_to_external_is_external(self):
        mod = _module_with_members(set())
        assert mod._corridor_classify_forward(EXT_C, EXT_D) == "external"

    def test_unknown_membership_defaults_to_external(self):
        """No fee_controller wired -> membership is unknown -> external,
        matching the fee gate's own fail-safe default."""
        mod = load_plugin_module()
        mod.fee_controller = None
        assert mod._corridor_classify_forward(HIVE_A, HIVE_B) == "external"

    def test_none_peer_ids_default_to_external(self):
        mod = _module_with_members({HIVE_A})
        assert mod._corridor_classify_forward(None, None) == "external"


class TestForwardEventRecordsCorridorFlow:
    def _settled_event(self, in_channel="100x1x0", out_channel="200x2x0"):
        return {
            "status": "settled",
            "in_channel": in_channel,
            "out_channel": out_channel,
            "in_msat": 1_000_000,
            "out_msat": 999_000,
            "fee_msat": 1_000,
            "received_time": 1_700_000_000,
            "resolved_time": 1_700_000_002,
        }

    def test_settled_forward_records_corridor_flow(self):
        mod = _module_with_members({HIVE_A, HIVE_B})
        db = MagicMock()
        mod.database = db

        def _resolve(scid):
            return {"100x1x0": HIVE_A, "200x2x0": HIVE_B}.get(scid)

        mod._resolve_scid_to_peer = MagicMock(side_effect=_resolve)

        mod._on_forward_event_impl(self._settled_event(), mod.plugin)

        db.corridor_flow_record.assert_called_once_with("internal_transit", 1_000, 1_000)

    def test_corridor_recording_missing_accessor_does_not_crash(self):
        """A database double without corridor_flow_record must not break
        forward-event handling (older partial deployments / test doubles)."""
        mod = _module_with_members({HIVE_A})
        db = MagicMock(spec=[
            "record_forward_and_reputation", "record_forward", "update_peer_reputation",
        ])
        mod.database = db
        mod._resolve_scid_to_peer = MagicMock(return_value=HIVE_A)

        # Must not raise.
        mod._on_forward_event_impl(self._settled_event(), mod.plugin)


class TestCorridorFlowDailyAccessor:
    def test_record_accumulates_same_day_same_klass(self):
        db = _make_db()
        db.corridor_flow_record("internal_transit", 1000, 10)
        db.corridor_flow_record("internal_transit", 2000, 20)

        summary = db.corridor_flow_summary(days=7)

        assert summary["by_klass"]["internal_transit"] == {
            "forwards": 2, "sats_forwarded": 3000, "fees_msat": 30,
        }
        assert summary["totals"] == {"forwards": 2, "sats_forwarded": 3000, "fees_msat": 30}

    def test_all_four_classes_present_even_when_empty(self):
        db = _make_db()
        summary = db.corridor_flow_summary(days=7)
        assert set(summary["by_klass"].keys()) == {
            "internal_transit", "edge_in", "edge_out", "external",
        }
        for row in summary["by_klass"].values():
            assert row == {"forwards": 0, "sats_forwarded": 0, "fees_msat": 0}

    def test_summary_excludes_days_outside_window(self):
        db = _make_db()
        conn = db._get_connection()
        old_day = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 30 * 86400))
        conn.execute(
            "INSERT INTO corridor_flow_daily (day, klass, forwards, sats_forwarded, fees_msat) "
            "VALUES (?, 'edge_in', 5, 5000, 50)",
            (old_day,),
        )
        db.corridor_flow_record("edge_in", 100, 1)

        summary = db.corridor_flow_summary(days=7)

        assert summary["by_klass"]["edge_in"]["forwards"] == 1
        assert summary["totals"]["forwards"] == 1

    def test_unknown_klass_is_ignored(self):
        db = _make_db()
        db.corridor_flow_record("bogus_klass", 100, 1)
        summary = db.corridor_flow_summary(days=7)
        assert summary["totals"] == {"forwards": 0, "sats_forwarded": 0, "fees_msat": 0}


class TestDashboardCorridorSection:
    def _load_report_surface_module(self):
        mod = load_plugin_module()
        mod.policy_manager = MagicMock()
        mod.profitability_analyzer = MagicMock()
        mod.database = MagicMock()
        return mod

    def test_dashboard_includes_mycelial_corridor_section(self):
        mod = self._load_report_surface_module()
        mod.profitability_analyzer.get_tlv.return_value = {"tlv_sats": 1_000_000}
        mod.profitability_analyzer.get_pnl_summary.return_value = {
            "net_profit_sats": 5, "operating_margin_pct": 10.0,
            "gross_revenue_sats": 10, "opex_sats": 5,
            "rebalance_cost_sats": 0, "closure_cost_sats": 0,
            "volume_sats": 100_000, "forward_count": 3,
        }
        mod.profitability_analyzer.calculate_roc.return_value = {"annualized_roc_pct": 1.0}
        mod.profitability_analyzer.identify_bleeders.return_value = []

        corridor_payload = {
            "days": 7,
            "by_klass": {
                "internal_transit": {"forwards": 10, "sats_forwarded": 500_000, "fees_msat": 0},
                "edge_in": {"forwards": 3, "sats_forwarded": 100_000, "fees_msat": 5000},
                "edge_out": {"forwards": 4, "sats_forwarded": 150_000, "fees_msat": 7000},
                "external": {"forwards": 1, "sats_forwarded": 10_000, "fees_msat": 500},
            },
            "totals": {"forwards": 18, "sats_forwarded": 760_000, "fees_msat": 12500},
        }
        mod.database.corridor_flow_summary.return_value = corridor_payload

        result = mod.revenue_dashboard(mod.plugin, window_days=30)

        assert "mycelial_corridor" in result
        section = result["mycelial_corridor"]
        assert section["window_days"] == 7
        assert section["by_klass"] == corridor_payload["by_klass"]
        assert section["totals"] == corridor_payload["totals"]
        assert section["fee_split_msat"]["internal"] == 0
        assert section["fee_split_msat"]["edge"] == 5000 + 7000 + 500
        mod.database.corridor_flow_summary.assert_called_once_with(days=7)

    def test_dashboard_omits_section_when_accessor_missing(self):
        mod = self._load_report_surface_module()
        mod.profitability_analyzer.get_tlv.return_value = {"tlv_sats": 0}
        mod.profitability_analyzer.get_pnl_summary.return_value = {}
        mod.profitability_analyzer.calculate_roc.return_value = {}
        mod.profitability_analyzer.identify_bleeders.return_value = []
        mod.database = MagicMock(spec=[
            "get_all_channel_states",  # anything, just no corridor_flow_summary
        ])

        result = mod.revenue_dashboard(mod.plugin, window_days=30)

        assert "mycelial_corridor" not in result


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
