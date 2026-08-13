"""Phase 1 wiring: econ shadow touchpoints in cl-revenue-ops.py.

Proves the fail-open contract at the plugin level: a broken or absent
shadow must never affect the fee cycle, and the RPC never raises.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


def _adjustments():
    return [
        SimpleNamespace(channel_id="100x1x0", peer_id="02" + "c" * 64,
                        old_fee_ppm=100, new_fee_ppm=250, reason="dts",
                        algorithm_values={}, reason_code="dts_pid_sample"),
        SimpleNamespace(channel_id="200x1x0", peer_id="02" + "d" * 64,
                        old_fee_ppm=300, new_fee_ppm=280, reason="dts",
                        algorithm_values={}, reason_code="dts_pid_sample"),
    ]


def _fee_module(shadow):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = MagicMock()
    mod.fee_controller.adjust_all_fees.return_value = _adjustments()
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        min_fee_ppm=10, max_fee_ppm=2000, fee_interval=1800)
    mod.data_service = None
    mod.econ_shadow = shadow
    return mod


def test_enabled_shadow_receives_cycle_adjustments():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    mod = _fee_module(shadow)
    mod.run_fee_adjustment()
    shadow.record_fee_intents.assert_called_once()
    args = shadow.record_fee_intents.call_args.args
    assert len(args[0]) == 2


def test_disabled_shadow_not_asked_to_record():
    shadow = MagicMock()
    shadow.enabled.return_value = False
    mod = _fee_module(shadow)
    mod.run_fee_adjustment()
    shadow.record_fee_intents.assert_not_called()


def test_raising_shadow_does_not_break_fee_cycle():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    shadow.record_fee_intents.side_effect = RuntimeError("shadow broke")
    mod = _fee_module(shadow)
    mod.run_fee_adjustment()  # must not raise
    mod.fee_controller.adjust_all_fees.assert_called_once()


def test_absent_shadow_is_harmless():
    mod = _fee_module(None)
    mod.run_fee_adjustment()  # must not raise
    mod.fee_controller.adjust_all_fees.assert_called_once()


def test_fee_cycle_does_not_own_reconciliation_schedule():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    mod = _fee_module(shadow)
    mod.database = MagicMock()
    mod.run_fee_adjustment()
    shadow.maybe_run_reconciliation.assert_not_called()


def test_scheduled_reconciliation_runs_without_fee_authority():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    mod = _fee_module(shadow)
    mod.database = MagicMock()
    mod.fee_authority_gate = MagicMock()
    mod._run_scheduled_reconciliation(now=1_754_006_401)
    shadow.maybe_run_reconciliation.assert_called_once_with(
        mod.database, 1_754_006_401)


def test_scheduled_reconciliation_passes_missing_database_for_skip_evidence():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    mod = _fee_module(shadow)
    mod.database = None
    mod._run_scheduled_reconciliation(now=1_754_006_401)
    shadow.maybe_run_reconciliation.assert_called_once_with(
        None, 1_754_006_401)


def test_raising_scheduled_sweep_is_contained():
    shadow = MagicMock()
    shadow.enabled.return_value = True
    shadow.maybe_run_reconciliation.side_effect = RuntimeError("boom")
    mod = _fee_module(shadow)
    mod.database = MagicMock()
    assert mod._run_scheduled_reconciliation(now=1_754_006_401) is None
    mod.plugin.log.assert_called_with(
        "reconciliation sweep skipped: boom", level="debug")


class TestSnapshotRpc:
    def test_shadow_none_reports_disabled(self):
        mod = load_plugin_module()
        mod.econ_shadow = None
        result = mod.revenue_econ_snapshot(mod.plugin)
        assert result["enabled"] is False

    def test_shadow_disabled_reports_disabled(self):
        mod = load_plugin_module()
        shadow = MagicMock()
        shadow.enabled.return_value = False
        mod.econ_shadow = shadow
        result = mod.revenue_econ_snapshot(mod.plugin)
        assert result["enabled"] is False

    def test_enabled_preview_returned(self):
        mod = load_plugin_module()
        shadow = MagicMock()
        shadow.enabled.return_value = True
        shadow.intents_recorded_total = 7
        shadow.build_snapshot_preview.return_value = (
            {"schema_name": "economic_snapshot"}, ["placeholder"])
        mod.econ_shadow = shadow
        mod.data_service = MagicMock()
        mod.data_service.get_peer_channels.return_value = {"channels": []}
        mod.profitability_analyzer = MagicMock()
        mod.profitability_analyzer.analyze_all_channels.return_value = {}
        mod.database = MagicMock()
        mod.database.get_budget_status.return_value = {
            "spent": 100, "reserved": 50, "total_committed": 150}
        mod.config = MagicMock()
        mod.config.snapshot.return_value = SimpleNamespace(
            daily_budget_sats=5000, receivable_ratio_target=0.3)
        result = mod.revenue_econ_snapshot(mod.plugin)
        assert result["enabled"] is True
        assert result["snapshot"] == {"schema_name": "economic_snapshot"}
        assert result["approximations"] == ["placeholder"]
        assert result["intents_recorded_total"] == 7
        budget = shadow.build_snapshot_preview.call_args.kwargs["budget"]
        assert budget == {"cap_sats": 5000, "reserved_sats": 50,
                          "spent_sats": 100}

    def test_rpc_never_raises(self):
        mod = load_plugin_module()
        shadow = MagicMock()
        shadow.enabled.return_value = True
        shadow.build_snapshot_preview.side_effect = RuntimeError("boom")
        mod.econ_shadow = shadow
        mod.data_service = None
        mod.profitability_analyzer = None
        mod.database = None
        mod.config = None
        result = mod.revenue_econ_snapshot(mod.plugin)
        assert result["enabled"] is True
        assert "error" in result or result.get("snapshot") is None


class TestReconcileRpc:
    def _mod(self, tmp_path):
        import os
        import tempfile
        from modules.database import Database
        from modules.econ_shadow import EconShadow

        mod = load_plugin_module()
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mod.database = Database(db_path, MagicMock())
        mod.database.initialize()
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(
            econ_shadow_enabled=True)
        cfg.db_path = db_path
        shadow = EconShadow(MagicMock(), cfg,
                            ledger_path=str(tmp_path / "ledger.db"))
        mod.database.spend_journal = shadow
        mod.econ_shadow = shadow
        return mod

    def test_disabled_reports_disabled(self):
        mod = load_plugin_module()
        mod.econ_shadow = None
        assert mod.revenue_econ_reconcile(mod.plugin)["enabled"] is False

    def test_dry_run_then_apply(self, tmp_path):
        mod = self._mod(tmp_path)
        # Journaled reserve, then settle with journaling DISABLED —
        # a real ledger_stale_reservation divergence.
        mod.database.reserve_spend(reservation_id="op-1", amount_sats=3,
                                   category="planner")
        mod.econ_shadow._config.snapshot.return_value = SimpleNamespace(
            econ_shadow_enabled=False)
        mod.database.mark_spend_reservation_spent("op-1")
        mod.econ_shadow._config.snapshot.return_value = SimpleNamespace(
            econ_shadow_enabled=True)

        dry = mod.revenue_econ_reconcile(mod.plugin)
        assert dry["enabled"] is True
        assert [d["kind"] for d in dry["divergences"]] == [
            "ledger_stale_reservation"]
        assert "applied" not in dry  # dry-run default

        applied = mod.revenue_econ_reconcile(mod.plugin, apply=True)
        assert applied["applied"] == 1
        clean = mod.revenue_econ_reconcile(mod.plugin)
        assert clean["divergences"] == []

    def test_rpc_never_raises(self):
        mod = load_plugin_module()
        shadow = MagicMock()
        shadow.enabled.return_value = True
        shadow.ledger_for_reconciliation.side_effect = RuntimeError("boom")
        mod.econ_shadow = shadow
        mod.database = None
        result = mod.revenue_econ_reconcile(mod.plugin)
        assert result["enabled"] is True and "error" in result
