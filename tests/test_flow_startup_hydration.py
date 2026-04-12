import pytest
from unittest.mock import MagicMock, patch

from tests.plugin_test_utils import load_plugin_module


NOW = 1_700_000_000


def _compute_start(last_forward_ts, flow_window_days):
    mod = load_plugin_module()
    return mod._compute_forward_hydration_start(last_forward_ts, flow_window_days, NOW)


@pytest.mark.parametrize(
    "flow_window_days, expected_days",
    [
        (7, 14),
        (21, 21),
    ],
)
def test_none_last_forward_ts_covers_window_or_fourteen_days(flow_window_days, expected_days):
    start = _compute_start(None, flow_window_days)

    assert start == NOW - (expected_days * 86400)


def test_non_empty_table_with_meaningful_gap_uses_bounded_overlap_start():
    # 30 days old with a 7-day flow window should be capped by the 15-day floor.
    last_forward_ts = NOW - (30 * 86400)

    start = _compute_start(last_forward_ts, 7)

    assert start == NOW - (15 * 86400)


def test_very_recent_last_forward_ts_returns_none():
    last_forward_ts = NOW - 30 * 60

    assert _compute_start(last_forward_ts, 7) is None


def test_startup_hydration_uses_helper_window_for_empty_table():
    mod = load_plugin_module()
    import modules.data_service as data_service_module
    import modules.rebalance_engine_v2 as rebalance_engine_module

    fake_db = MagicMock()
    fake_db.get_latest_forward_timestamp.return_value = None
    fake_db.bulk_insert_forwards = MagicMock()
    fake_db.cleanup_stale_reservations.return_value = 0
    fake_db.has_recent_connection_history.return_value = True
    fake_db.get_peers.return_value = {"peers": []}

    fake_safe_plugin = MagicMock()
    fake_safe_plugin.rpc = MagicMock()
    fake_safe_plugin.rpc._executor = MagicMock()
    fake_safe_plugin.rpc._async_executor = MagicMock()

    fake_data_service = MagicMock()
    fake_data_service.list_plugins.return_value = {"plugins": []}
    fake_data_service.get_forwards.return_value = {
        "forwards": [
            {
                "in_channel": "100x1x0",
                "out_channel": "200x2x0",
                "in_msat": "1000000msat",
                "out_msat": "900000msat",
                "fee_msat": "100000msat",
                "received_time": NOW - int(14.5 * 86400),
                "resolved_time": NOW - int(14.5 * 86400) + 30,
            }
        ]
    }

    class _FakeComponent:
        def __init__(self, *args, **kwargs):
            self.hive_hints = None
            self.data_service = None

        def __getattr__(self, name):
            return MagicMock()

    class _FakeRebalancer(_FakeComponent):
        def set_profitability_analyzer(self, *args, **kwargs):
            return None

        def set_capacity_planner(self, *args, **kwargs):
            return None

        def set_capex_engine(self, *args, **kwargs):
            return None

    class _FakeCapacityPlanner(_FakeComponent):
        def set_capital_efficiency(self, *args, **kwargs):
            return None

        def set_capex_engine(self, *args, **kwargs):
            return None

    class _FakeCapexEngine(_FakeComponent):
        pass

    class _FakeBoltzManager(_FakeComponent):
        enabled = False

        def set_capex_engine(self, *args, **kwargs):
            return None

    options = {}
    for name, spec in mod.plugin.options.items():
        default = spec.get("default", "")
        options[name] = "" if default is None else str(default)

    options["revenue-ops-flow-window-days"] = "7"
    options["revenue-ops-db-path"] = ":memory:"
    options["revenue-ops-boltz-enabled"] = "false"

    with (
        patch.object(mod, "Database", return_value=fake_db),
        patch.object(mod, "ThreadSafePluginProxy", return_value=fake_safe_plugin),
        patch.object(mod, "PolicyManager", return_value=_FakeComponent()),
        patch.object(mod, "ChannelProfitabilityAnalyzer", return_value=_FakeComponent()),
        patch.object(mod, "FlowAnalyzer", return_value=_FakeComponent()),
        patch.object(mod, "CapacityPlanner", return_value=_FakeCapacityPlanner()),
        patch.object(mod, "FeeController", return_value=_FakeComponent()),
        patch.object(mod, "EVRebalancer", return_value=_FakeRebalancer()),
        patch.object(mod, "HiveHintAdapter", return_value=_FakeComponent()),
        patch.object(mod, "HiveRouter", return_value=_FakeComponent()),
        patch.object(mod, "CapitalEfficiencyAnalyzer", return_value=_FakeComponent()),
        patch.object(mod, "CapexBudgetEngine", return_value=_FakeCapexEngine()),
        patch.object(mod, "BoltzCliManager", return_value=_FakeBoltzManager()),
        patch.object(mod.Config, "load_overrides", return_value=[]),
        patch.object(mod, "_start_background_tasks", return_value=None, create=True),
        patch.object(data_service_module, "DataService", return_value=fake_data_service),
        patch.object(rebalance_engine_module, "RebalanceEngine", return_value=_FakeComponent()),
        patch.object(mod.time, "time", return_value=NOW),
    ):
        mod.init(options, {}, mod.plugin)

    fake_db.bulk_insert_forwards.assert_not_called()
