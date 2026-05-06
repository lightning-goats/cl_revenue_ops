import pytest
from unittest.mock import MagicMock
from pathlib import Path
from types import SimpleNamespace
from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


def test_public_runtime_keys_are_safety_only():
    cfg = Config()

    assert cfg.public_runtime_keys() == [
        "paused",
        "daily_budget_sats",
        "min_fee_ppm",
        "max_fee_ppm",
        "fee_profile",
        "fee_market_boundary_enabled",
        "fee_market_boundary_min_competitors",
        "fee_market_boundary_margin_ppm",
        "fee_market_boundary_margin_ratio",
        "fee_market_boundary_max_downshift_ratio",
        "fee_market_boundary_cache_seconds",
        "planner_enabled",
        "planner_dry_run",
        "planner_execute_closes",
        "planner_max_opens_per_cycle",
        "planner_max_closes_per_cycle",
        "planner_min_annual_roi_pct",
        "capex_probability_budget_bonus",
    ]


def test_internal_knobs_are_not_public():
    cfg = Config()

    assert "enable_vegas_reflex" not in cfg.public_runtime_keys()
    assert "thompson_prior_std_fee" not in cfg.public_runtime_keys()
    assert "askrene_layers" not in cfg.public_runtime_keys()
    assert "hive_hints_enabled" not in cfg.public_runtime_keys()


def test_public_runtime_dict_returns_only_public_keys():
    cfg = Config(paused=True, daily_budget_sats=1200, min_fee_ppm=15, max_fee_ppm=2500)

    assert cfg.public_runtime_dict() == {
        "paused": True,
        "daily_budget_sats": 1200,
        "min_fee_ppm": 15,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "planner_enabled": False,
        "planner_dry_run": False,
        "planner_execute_closes": False,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 0,
        "planner_min_annual_roi_pct": 1.0,
        "capex_probability_budget_bonus": 0.0,
    }


def test_runtime_key_classification_distinguishes_public_deprecated_and_internal():
    cfg = Config()

    assert cfg.classify_runtime_key("paused") == "public"
    assert cfg.classify_runtime_key("enable_vegas_reflex") == "internal"
    assert cfg.classify_runtime_key("askrene_layers") == "internal"
    assert cfg.classify_runtime_key("hive_hints_enabled") == "internal"
    assert cfg.classify_runtime_key("dry_run") == "internal"


def _load_operator_surface_module():
    mod = load_plugin_module()
    mod.database = MagicMock()
    mod.config = Config(
        paused=True,
        daily_budget_sats=1200,
        min_fee_ppm=15,
        max_fee_ppm=2500,
    )
    return mod


def _load_policy_surface_module():
    mod = load_plugin_module()
    mod.policy_manager = MagicMock()
    return mod


def _load_report_surface_module():
    mod = load_plugin_module()
    mod.policy_manager = MagicMock()
    mod.profitability_analyzer = MagicMock()
    mod.database = MagicMock()
    return mod


def _default_plugin_options(mod):
    return {
        name: registration["default"]
        for name, registration in mod.plugin.options.items()
        if "default" in registration
    }


def _run_init_with_stubbed_dependencies(
    mod,
    monkeypatch,
    option_overrides=None,
    db_overrides=None,
    rpc_mutator=None,
    listconfigs_payload=None,
):
    mod.shutdown_event.clear()

    options = _default_plugin_options(mod)
    if option_overrides:
        options.update(option_overrides)

    fake_db = MagicMock()
    fake_db.initialize.return_value = None
    fake_db.cleanup_stale_reservations.return_value = 0
    fake_db.get_latest_forward_timestamp.return_value = None
    fake_db.bulk_insert_forwards.return_value = 0
    fake_db.has_recent_connection_history.return_value = False
    fake_db.record_connection_event.return_value = None
    fake_db.get_all_config_overrides.return_value = db_overrides or {}
    fake_db.get_config_version.return_value = 1 if db_overrides else 0

    fake_rpc = MagicMock()
    fake_rpc.plugin.return_value = {"plugins": []}
    fake_rpc.listplugins.return_value = {"plugins": []}
    fake_rpc.listforwards.return_value = {"forwards": []}
    fake_rpc.listpeers.return_value = {"peers": []}
    fake_rpc.listconfigs.return_value = listconfigs_payload or {"configs": {}}
    if rpc_mutator is not None:
        rpc_mutator(fake_rpc)

    fake_proxy = MagicMock()
    fake_proxy.rpc = fake_rpc
    fake_proxy._executor = MagicMock()
    fake_proxy._async_executor = MagicMock()
    mod._test_fake_rpc = fake_rpc

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(mod, "ThreadSafePluginProxy", lambda plugin: fake_proxy)
    monkeypatch.setattr(mod, "Database", lambda *args, **kwargs: fake_db)
    monkeypatch.setattr(mod, "PolicyManager", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "ChannelProfitabilityAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "FlowAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "CapacityPlanner", lambda *args, **kwargs: MagicMock(
        execute_cycle=MagicMock(return_value={"skipped": True, "reason": "noop"})
    ))
    monkeypatch.setattr(mod, "FeeController", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "EVRebalancer", lambda *args, **kwargs: MagicMock(
        set_profitability_analyzer=MagicMock(),
        set_capacity_planner=MagicMock(),
    ))
    monkeypatch.setattr(mod, "BoltzCliManager", lambda *args, **kwargs: MagicMock(enabled=False))
    monkeypatch.setattr(mod.threading, "Thread", DummyThread)

    mod.init(options, {}, mod.plugin)
    return mod.config


def test_planner_execute_closes_plugin_option_defaults_false():
    mod = load_plugin_module()
    snapshot = Config().snapshot()

    assert "revenue-ops-planner-execute-closes" in mod.plugin.options
    assert mod.plugin.options["revenue-ops-planner-execute-closes"]["default"] == "false"
    assert snapshot.planner_execute_closes is False


def test_planner_cycle_limit_defaults_match_config():
    mod = load_plugin_module()
    cfg = Config()
    snapshot = cfg.snapshot()

    assert cfg.planner_max_opens_per_cycle == 1
    assert cfg.planner_max_closes_per_cycle == 0
    assert snapshot.planner_max_opens_per_cycle == 1
    assert snapshot.planner_max_closes_per_cycle == 0
    assert snapshot.planner_min_annual_roi_pct == 1.0
    assert mod.plugin.options["revenue-ops-planner-max-opens-per-cycle"]["default"] == "1"
    assert mod.plugin.options["revenue-ops-planner-max-closes-per-cycle"]["default"] == "0"
    assert mod.plugin.options["revenue-ops-planner-min-annual-roi-pct"]["default"] == "1.0"


def test_planner_execute_closes_option_is_parsed_during_init(monkeypatch):
    mod = load_plugin_module()
    cfg = _run_init_with_stubbed_dependencies(
        mod,
        monkeypatch,
        {"revenue-ops-planner-execute-closes": "true"},
    )

    assert cfg.planner_execute_closes is True


def test_planner_cycle_limits_are_parsed_during_init(monkeypatch):
    mod = load_plugin_module()
    cfg = _run_init_with_stubbed_dependencies(
        mod,
        monkeypatch,
        {
            "revenue-ops-planner-max-opens-per-cycle": "3",
            "revenue-ops-planner-max-closes-per-cycle": "2",
        },
    )

    assert cfg.planner_max_opens_per_cycle == 3
    assert cfg.planner_max_closes_per_cycle == 2


def test_planner_min_annual_roi_option_is_parsed_during_init(monkeypatch):
    mod = load_plugin_module()
    cfg = _run_init_with_stubbed_dependencies(
        mod,
        monkeypatch,
        {"revenue-ops-planner-min-annual-roi-pct": "2.5"},
    )

    assert cfg.planner_min_annual_roi_pct == 2.5


def test_init_wires_rebalancer_back_into_capacity_planner(monkeypatch):
    mod = load_plugin_module()
    mod.shutdown_event.clear()

    options = _default_plugin_options(mod)
    fake_db = MagicMock()
    fake_db.initialize.return_value = None
    fake_db.cleanup_stale_reservations.return_value = 0
    fake_db.get_latest_forward_timestamp.return_value = None
    fake_db.bulk_insert_forwards.return_value = 0
    fake_db.has_recent_connection_history.return_value = False
    fake_db.record_connection_event.return_value = None
    fake_db.get_all_config_overrides.return_value = {}
    fake_db.get_config_version.return_value = 0

    fake_rpc = MagicMock()
    fake_rpc.plugin.return_value = {"plugins": []}
    fake_rpc.listplugins.return_value = {"plugins": []}
    fake_rpc.listforwards.return_value = {"forwards": []}
    fake_rpc.listpeers.return_value = {"peers": []}
    fake_rpc.listconfigs.return_value = {"configs": {}}

    fake_proxy = MagicMock()
    fake_proxy.rpc = fake_rpc
    fake_proxy._executor = MagicMock()
    fake_proxy._async_executor = MagicMock()

    planner_mock = MagicMock()
    planner_mock.rebalancer = None
    rebalancer_mock = MagicMock(
        set_profitability_analyzer=MagicMock(),
        set_capacity_planner=MagicMock(),
    )

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(mod, "ThreadSafePluginProxy", lambda plugin: fake_proxy)
    monkeypatch.setattr(mod, "Database", lambda *args, **kwargs: fake_db)
    monkeypatch.setattr(mod, "PolicyManager", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "ChannelProfitabilityAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "FlowAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "CapacityPlanner", lambda *args, **kwargs: planner_mock)
    monkeypatch.setattr(mod, "FeeController", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "EVRebalancer", lambda *args, **kwargs: rebalancer_mock)
    monkeypatch.setattr(mod, "BoltzCliManager", lambda *args, **kwargs: MagicMock(enabled=False))
    monkeypatch.setattr(mod.threading, "Thread", DummyThread)

    mod.init(options, {}, mod.plugin)

    assert planner_mock.rebalancer is rebalancer_mock


def test_init_logs_native_route_rebalancing(monkeypatch):
    mod = load_plugin_module()
    _run_init_with_stubbed_dependencies(mod, monkeypatch)

    messages = [call.args[0] for call in mod.plugin.log.call_args_list if call.args]

    assert any("native route execution" in message.lower() for message in messages)
    assert not any("getroute + sendpay" in message for message in messages)


def test_revenue_rebalance_debug_reports_segment_hint_influence():
    mod = load_plugin_module()
    mod.config = Config()
    mod.data_service = MagicMock()
    mod.data_service.get_funds.return_value = {"outputs": [], "channels": []}
    mod.database = MagicMock()
    mod.database.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "stale_reservations": 0,
        "job_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    mod.database.get_all_channel_states.return_value = []
    mod._total_cost_budget_status = MagicMock(
        return_value={
            "effective_budget_sats": 1000,
            "remaining_sats": 1000,
            "actual_spent_by_category": {},
            "reserved_by_category": {},
        }
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    mod.rebalancer = SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
    )
    mod.hive_hints = MagicMock()
    mod.hive_hints.get_status.return_value = {
        "snapshot_fresh": True,
        "snapshot_age_seconds": 10,
        "hints_count": 2,
    }
    mod.hive_hints.get_segment_scores.return_value = [
        {
            "short_channel_id": "123x1x0",
            "direction": 1,
            "amount_bucket_sats": 250_000,
            "success_score": 0.0,
            "failure_score": 0.7,
            "net_utility": -0.7,
            "confidence": 0.8,
            "observer_count": 2,
            "last_observed_at": 1_700_000_000,
        }
    ]

    result = mod.revenue_rebalance_debug(mod.plugin)

    mod.hive_hints.poll.assert_called_once()
    assert result["hive_hints"]["status_refresh_attempted"] is True
    assert result["hive_hints"]["status_refresh_ok"] is True
    assert result["hive_hints"]["segment_scores_count"] == 1
    assert result["hive_hints"]["segment_scores"][0]["short_channel_id"] == "123x1x0"


def test_revenue_rebalance_debug_exposes_hive_hint_refresh_diagnostics():
    mod = load_plugin_module()
    mod.config = Config()
    mod.data_service = MagicMock()
    mod.data_service.get_funds.return_value = {"outputs": [], "channels": []}
    mod.database = MagicMock()
    mod.database.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "stale_reservations": 0,
        "job_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    mod.database.get_all_channel_states.return_value = []
    mod._total_cost_budget_status = MagicMock(
        return_value={
            "effective_budget_sats": 1000,
            "remaining_sats": 1000,
            "actual_spent_by_category": {},
            "reserved_by_category": {},
        }
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    mod.rebalancer = SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
    )
    mod.hive_hints = MagicMock()
    mod.hive_hints.refresh_status_for_debug.return_value = {
        "refresh_needed": True,
        "refresh_attempted": True,
        "refresh_result": "refreshed_from_datastore",
        "cache": {
            "source": "hive_export_rpc",
            "age_seconds": 391,
            "effective_ttl_seconds": 300,
            "usable": False,
        },
        "cache_after_refresh": {
            "source": "datastore",
            "age_seconds": 5,
            "effective_ttl_seconds": 300,
            "usable": True,
        },
        "live_datastore": {
            "queried": True,
            "generation": 5915,
            "age_seconds": 5,
            "ttl_seconds": 300,
            "usable": True,
        },
        "live_hive_export": {
            "queried": False,
            "usable": False,
        },
        "fallback": {
            "needed": False,
            "reason": "",
            "used": False,
            "used_source": "",
            "stale_fallback_used": False,
        },
    }
    mod.hive_hints.get_status.return_value = {
        "snapshot_fresh": True,
        "snapshot_usable": True,
        "snapshot_age_seconds": 5,
        "hints_count": 2,
    }
    mod.hive_hints.get_segment_scores.return_value = []

    result = mod.revenue_rebalance_debug(mod.plugin)

    mod.hive_hints.refresh_status_for_debug.assert_called_once()
    mod.hive_hints.poll.assert_not_called()
    hive_status = result["hive_hints"]
    assert hive_status["status_refresh_needed"] is True
    assert hive_status["status_refresh_attempted"] is True
    assert hive_status["status_refresh_result"] == "refreshed_from_datastore"
    assert hive_status["cache"]["age_seconds"] == 391
    assert hive_status["cache"]["source"] == "hive_export_rpc"
    assert hive_status["cache_after_refresh"]["source"] == "datastore"
    assert hive_status["live_datastore"]["generation"] == 5915
    assert hive_status["live_hive_export"]["queried"] is False
    assert hive_status["fallback"]["needed"] is False


def test_revenue_rebalance_debug_includes_last_cycle_score_breakdown():
    mod = load_plugin_module()
    mod.config = Config()
    mod.data_service = MagicMock()
    mod.data_service.get_funds.return_value = {"outputs": [], "channels": []}
    mod.database = MagicMock()
    mod.database.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "stale_reservations": 0,
        "job_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    mod.database.get_all_channel_states.return_value = []
    mod._total_cost_budget_status = MagicMock(
        return_value={
            "effective_budget_sats": 1000,
            "remaining_sats": 1000,
            "actual_spent_by_category": {},
            "reserved_by_category": {},
        }
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    engine_debug = {
        "summary": {
            "considered_pairs": 1,
            "selected_pairs": 0,
            "skipped_pairs": 1,
            "execution_count": 0,
            "execution_success_count": 0,
            "valuable_channel_count": 2,
            "total_remaining_budget_sats": 1000,
        },
        "considered_candidates": [
            {
                "source_channel_id": "100x1x0",
                "dest_channel_id": "200x2x0",
                "score_decomposition": {
                    "p_success": 0.6,
                    "expected_future_value": 1.0,
                    "expected_fee": 1.0,
                    "source_opportunity_cost": 0.1,
                    "failure_penalty": 0.0,
                    "capital_risk_penalty": 1.0,
                    "final_score": -1.5,
                    "beats_do_nothing": False,
                    "rejection_reason": "route_over_budget",
                },
            }
        ],
        "selected_candidates": [],
        "skipped": [],
        "executions": [],
    }
    mod.rebalancer = SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
        get_last_decision_summary=lambda: {
            "action": "hold",
            "reason": "no_rebalance_candidates",
            "dominant_input": "rebalance_engine",
            "safety_block": False,
            "budget_blocked": False,
        },
        rebalance_engine_v2=SimpleNamespace(
            get_last_cycle_debug=lambda max_candidates=10: engine_debug
        ),
    )
    mod.hive_hints = MagicMock()
    mod.hive_hints.get_status.return_value = {
        "snapshot_fresh": True,
        "snapshot_age_seconds": 5,
        "hints_count": 0,
    }
    mod.hive_hints.get_segment_scores.return_value = []

    result = mod.revenue_rebalance_debug(mod.plugin)

    assert result["last_decision"]["reason"] == "no_rebalance_candidates"
    assert result["last_cycle"]["summary"]["considered_pairs"] == 1
    assert (
        result["last_cycle"]["considered_candidates"][0]["score_decomposition"]["rejection_reason"]
        == "route_over_budget"
    )


def test_hive_hint_plugin_options_are_registered():
    mod = load_plugin_module()

    assert mod.plugin.options["revenue-ops-hive-hints-enabled"]["default"] == "true"
    assert mod.plugin.options["revenue-ops-hive-hints-ttl"]["default"] == "0"


def test_rebalance_tuning_plugin_options_are_dynamic():
    mod = load_plugin_module()

    for option in (
        "revenue-ops-rebalance-hold-margin",
        "revenue-ops-pair-fee-cap-ppm",
        "revenue-ops-hive-rebalance-bootstrap-budget-sats",
    ):
        assert mod.plugin.options[option]["dynamic"] is True
        assert mod.plugin.options[option]["on_change"] is mod._on_rebalance_tuning_change


def test_rebalance_tuning_setconfig_updates_live_config():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(
        rebalance_hold_margin=0.0,
        pair_fee_cap_ppm=1000,
        hive_rebalance_bootstrap_budget_sats=300,
    )

    mod._on_rebalance_tuning_change(
        mod.plugin,
        "revenue-ops-rebalance-hold-margin",
        "0.25",
    )
    mod._on_rebalance_tuning_change(
        mod.plugin,
        "revenue-ops-pair-fee-cap-ppm",
        "750",
    )
    mod._on_rebalance_tuning_change(
        mod.plugin,
        "revenue-ops-hive-rebalance-bootstrap-budget-sats",
        "125",
    )

    assert mod.config.rebalance_hold_margin == 0.25
    assert mod.config.pair_fee_cap_ppm == 750
    assert mod.config.hive_rebalance_bootstrap_budget_sats == 125


def test_rebalance_tuning_setconfig_rejects_out_of_range_values():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(rebalance_hold_margin=0.0)

    with pytest.raises(ValueError, match="<= 1.0"):
        mod._on_rebalance_tuning_change(
            mod.plugin,
            "revenue-ops-rebalance-hold-margin",
            "1.25",
        )


def test_askrene_layer_option_defaults_to_hive_fleet_bias():
    mod = load_plugin_module()

    assert mod.plugin.options["revenue-ops-askrene-layers"]["default"] == "hive-fleet"
    assert "askrene getroutes" in mod.plugin.options["revenue-ops-rebalance-router"]["description"]
    assert "cl-hive bias" in mod.plugin.options["revenue-ops-askrene-layers"]["description"]
    assert "standalone" in mod.plugin.options["revenue-ops-askrene-layers"]["description"]


def test_revenue_config_list_mutable_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "list-mutable")

    assert result["mutable_keys"] == [
        "capex_probability_budget_bonus",
        "daily_budget_sats",
        "fee_market_boundary_cache_seconds",
        "fee_market_boundary_enabled",
        "fee_market_boundary_margin_ppm",
        "fee_market_boundary_margin_ratio",
        "fee_market_boundary_max_downshift_ratio",
        "fee_market_boundary_min_competitors",
        "fee_profile",
        "max_fee_ppm",
        "min_fee_ppm",
        "paused",
        "planner_dry_run",
        "planner_enabled",
        "planner_execute_closes",
        "planner_max_closes_per_cycle",
        "planner_max_opens_per_cycle",
        "planner_min_annual_roi_pct",
    ]
    assert result["count"] == 18


def test_revenue_config_get_without_key_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "get")

    assert result["config"] == {
        "paused": True,
        "daily_budget_sats": 1200,
        "min_fee_ppm": 15,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "planner_enabled": False,
        "planner_dry_run": False,
        "planner_execute_closes": False,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 0,
        "planner_min_annual_roi_pct": 1.0,
        "capex_probability_budget_bonus": 0.0,
    }


def test_revenue_config_rejects_internal_knob_updates():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "set", "enable_vegas_reflex", "false")

    assert result["error"].startswith(
        "Key 'enable_vegas_reflex' is not a public runtime control"
    )


def test_revenue_config_get_internal_key_marks_it_internal():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "get", "enable_vegas_reflex")

    assert result["classification"] == "internal"
    assert result["warning"].startswith(
        "Key 'enable_vegas_reflex' is not a public runtime control"
    )


def test_revenue_config_allows_public_set_updates():
    mod = _load_operator_surface_module()
    mod.config.update_runtime = MagicMock(
        return_value={
            "status": "success",
            "old_value": True,
            "new_value": False,
            "version": 1,
        }
    )

    result = mod.revenue_config(mod.plugin, "set", "paused", "false")

    assert result["status"] == "success"
    mod.config.update_runtime.assert_called_once_with(mod.database, "paused", "false")


def test_revenue_config_rejects_internal_knob_resets():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "reset", "enable_vegas_reflex")

    assert result["error"].startswith(
        "Key 'enable_vegas_reflex' is not a public runtime control"
    )


def test_revenue_config_allows_public_resets():
    mod = _load_operator_surface_module()
    mod.database.delete_config_override.return_value = True

    result = mod.revenue_config(mod.plugin, "reset", "paused")

    assert result["status"] == "success"
    assert "removed" in result["message"]


def test_revenue_config_no_args_returns_public_config_dict():
    """Caller invokes ``lightning-cli revenue-config`` (or HTTP equivalent
    with empty params) and gets the public config snapshot back instead of
    a TypeError. Regression guard for the 22 traceback reports observed on
    nexus-01 between Feb-Apr 2026 from rune_id 17 hitting the action-based
    API with empty params."""
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin)

    assert "config" in result
    assert result["config"]["paused"] is True
    assert result["config"]["daily_budget_sats"] == 1200


def test_revenue_set_fee_accepts_full_channel_id_hash():
    mod = load_plugin_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.set_channel_fee.return_value = {
        "success": True,
        "channel_id": "123x456x0",
        "fee_ppm": 125,
    }
    mod.config = Config(min_fee_ppm=10, max_fee_ppm=5000)
    mod.force_rate_limiter = MagicMock()

    full_channel_id = "ad" * 32
    result = mod.revenue_set_fee(mod.plugin, full_channel_id, 125)

    assert result["status"] == "success"
    mod.fee_controller.set_channel_fee.assert_called_once_with(
        full_channel_id,
        125,
        manual=True,
        enforce_limits=True,
    )


def test_revenue_set_fee_returns_error_when_controller_fails():
    mod = load_plugin_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.set_channel_fee.return_value = {
        "success": False,
        "channel_id": "123x456x0",
        "fee_ppm": 125,
        "message": "Channel 123x456x0 not found",
    }
    mod.config = Config(min_fee_ppm=10, max_fee_ppm=5000)
    mod.force_rate_limiter = MagicMock()

    result = mod.revenue_set_fee(mod.plugin, "123x456x0", 125)

    assert result["status"] == "error"
    assert result["success"] is False
    assert result["error"] == "Channel 123x456x0 not found"


def test_revenue_analyze_normalizes_colon_scid_before_flow_analysis():
    mod = load_plugin_module()
    mod.flow_analyzer = MagicMock()
    mod.flow_analyzer.analyze_channel.return_value = SimpleNamespace(
        to_dict=lambda: {"channel_id": "100x1x0"}
    )

    result = mod.revenue_analyze(mod.plugin, "100:1:0")

    mod.flow_analyzer.analyze_channel.assert_called_once_with("100x1x0")
    assert result["channel"] == "100x1x0"


def test_flow_analyzer_get_channel_matches_colon_form_scid():
    from modules.flow_analysis import FlowAnalyzer

    analyzer = FlowAnalyzer(MagicMock(), MagicMock(), MagicMock())
    analyzer._get_channels = MagicMock(
        return_value=[
            {
                "short_channel_id": "100x1x0",
                "channel_id": "100x1x0",
                "peer_id": "02" + "a" * 64,
            }
        ]
    )

    channel = analyzer._get_channel("100:1:0")

    assert channel is not None
    assert channel["short_channel_id"] == "100x1x0"


def test_revenue_config_swallows_unknown_kwargs_from_wrapped_form_body():
    """Some HTTP clients form-encode a JSON body and end up dispatching the
    method with a single unrecognized key like
    ``{'{"action":"get"}': ''}``. The handler must accept and ignore
    extra keys rather than 500-ing on TypeError. Regression guard for the
    same nexus-01 traceback class as the no-args case."""
    mod = _load_operator_surface_module()

    result = mod.revenue_config(
        mod.plugin,
        **{'{"action":"get"}': ""},
    )

    assert "config" in result


def test_revenue_policy_no_args_returns_policy_list():
    """``revenue-policy`` with no action defaults to listing all policies
    rather than raising TypeError. Mirrors the revenue-config defaulting
    so external monitoring callers don't pollute the log on probe."""
    mod = _load_policy_surface_module()
    mod.policy_manager.list_policies.return_value = []

    result = mod.revenue_policy(mod.plugin)

    assert "policies" in result or result.get("count", 0) == 0


def test_revenue_report_peer_returns_aggregated_profitability_and_all_flow_states():
    mod = _load_report_surface_module()
    peer_id = "02" + "a" * 64
    mod.policy_manager.get_policy.return_value = SimpleNamespace(
        to_dict=lambda: {"peer_id": peer_id, "strategy": "dynamic"}
    )
    mod.profitability_analyzer.get_profitability_by_peer.return_value = {
        "peer_id": peer_id,
        "channel_count": 2,
        "aggregate": {"net_profit_sats": 5},
        "channels": [{"channel_id": "100x1x0"}, {"channel_id": "200x2x0"}],
    }
    mod.database.get_all_channel_states.return_value = [
        {"peer_id": peer_id, "channel_id": "200x2x0", "state": "sink"},
        {"peer_id": peer_id, "channel_id": "100x1x0", "state": "source"},
        {"peer_id": "02" + "b" * 64, "channel_id": "300x3x0", "state": "balanced"},
    ]

    result = mod.revenue_report(mod.plugin, "peer", peer_id=peer_id)

    assert result["type"] == "peer"
    assert result["profitability"]["channel_count"] == 2
    assert [state["channel_id"] for state in result["flow_states"]] == [
        "100x1x0",
        "200x2x0",
    ]
    assert result["flow_state"] is None


def test_revenue_dashboard_bleeder_warning_uses_channel_id():
    mod = _load_report_surface_module()
    mod.profitability_analyzer.get_tlv.return_value = {"tlv_sats": 1_000_000}
    mod.profitability_analyzer.get_pnl_summary.return_value = {
        "net_profit_sats": -5,
        "operating_margin_pct": -50.0,
        "gross_revenue_sats": 10,
        "opex_sats": 15,
        "rebalance_cost_sats": 15,
        "closure_cost_sats": 0,
        "volume_sats": 100_000,
        "forward_count": 3,
    }
    mod.profitability_analyzer.calculate_roc.return_value = {
        "annualized_roc_pct": -1.0,
    }
    mod.profitability_analyzer.identify_bleeders.return_value = [{
        "channel_id": "123x1x0",
        "rebalance_cost_sats": 15,
        "revenue_sats": 10,
    }]

    result = mod.revenue_dashboard(mod.plugin, window_days=30)

    assert result["warnings"] == [
        "Channel 123x1x0 is bleeding: Spent 15 sats rebalancing, earned 10 sats."
    ]


def test_total_cost_budget_excludes_canonical_open_close_from_generic_spend():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(
        reservation_timeout_hours=4,
        daily_budget_sats=2000,
    )
    mod.database = MagicMock()
    mod.database.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 72,
        "reserved_24h_sats": 23,
        "spent_by_category": {
            "channel_open": 50,
            "channel_close": 15,
            "misc_ops": 7,
        },
        "reserved_by_category": {
            "channel_open": 20,
            "misc_ops": 3,
        },
    }
    mod.database.get_total_routing_revenue.return_value = 500
    mod.database.get_opening_costs_since.return_value = 100
    mod.database.get_closure_costs_since.return_value = 30
    mod._rebalance_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 11, "reserved_24h_sats": 2}
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 5, "reserved_24h_sats": 4}
    )

    result = mod._total_cost_budget_status(window_hours=24)

    assert result["actual_spent_by_category"] == {
        "rebalance": 11,
        "boltz": 5,
        "open": 100,
        "close": 30,
        "ledger": 7,
    }
    assert result["actual_spent_sats"] == 153
    assert result["reserved_by_category"] == {
        "rebalance": 2,
        "boltz": 4,
        "ledger": 23,
    }
    assert result["reserved_sats"] == 29
    assert result["components"]["generic_ledger"]["spent_24h_sats"] == 7
    assert result["components"]["generic_ledger"]["counted_spent_categories"] == {"misc_ops": 7}
    assert result["components"]["generic_ledger"]["excluded_spent_categories"] == {
        "channel_open": 50,
        "channel_close": 15,
    }


def test_revenue_status_operator_controls_hide_internal_knob_dump():
    mod = _load_operator_surface_module()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []

    result = mod.revenue_status(mod.plugin)

    assert result["operator_controls"]["values"] == {
        "paused": True,
        "daily_budget_sats": 1200,
        "min_fee_ppm": 15,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "planner_enabled": False,
        "planner_dry_run": False,
        "planner_execute_closes": False,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 0,
        "planner_min_annual_roi_pct": 1.0,
        "capex_probability_budget_bonus": 0.0,
    }
    assert "config" not in result
    assert "hive_hints" not in result
    assert "hive" not in result


@pytest.mark.parametrize(
    ("action", "peer_id", "kwargs"),
    [
        ("set", "02" + "a" * 64, {"strategy": "static", "fee_ppm": 500}),
        ("delete", "02" + "a" * 64, {}),
        ("tag", "02" + "a" * 64, {"tag": "whale"}),
        ("untag", "02" + "a" * 64, {"tag": "whale"}),
        ("batch", None, {"updates": [{"peer_id": "02" + "a" * 64, "strategy": "dynamic"}]}),
    ],
)
def test_revenue_policy_mutations_are_deprecated_for_normal_operator_use(action, peer_id, kwargs):
    mod = _load_policy_surface_module()

    result = mod.revenue_policy(mod.plugin, action, peer_id, **kwargs)

    assert result["error"].startswith(
        f"revenue-policy {action} is deprecated for normal operator use"
    )


def test_revenue_policy_list_remains_available_for_transition_diagnostics():
    mod = _load_policy_surface_module()
    policy = MagicMock()
    policy.to_dict.return_value = {"peer_id": "02" + "a" * 64, "strategy": "dynamic"}
    mod.policy_manager.get_all_policies.return_value = [policy]

    result = mod.revenue_policy(mod.plugin, "list")

    assert result["count"] == 1


def test_readme_examples_describe_current_hive_surfaces():
    readme = Path("README.md").read_text()

    assert "paused" in readme
    assert "daily_budget_sats" in readme
    assert "min_fee_ppm" in readme
    assert "max_fee_ppm" in readme
    assert "decision explainability" in readme
    assert "revenue-config set enable_vegas_reflex false" not in readme
    assert "lightning-cli revenue-policy set" not in readme
    assert "cl-mycelium" in readme
    assert "hive-hints" in readme
    assert "mycelium/hive hint contract" in readme
    assert "revenue-hive-hints-status" in readme


def test_claude_docs_describe_current_hive_hint_surface():
    claude = Path("CLAUDE.md").read_text()

    assert "cl-hive" in claude
    assert "hive-hints" in claude


def test_readme_states_planner_closes_are_recommendation_only_by_default():
    readme = Path("README.md").read_text()

    assert "Planner closes are recommendation-only by default." in readme
    assert "revenue-ops-planner-execute-closes=true" in readme


def test_readme_describes_boltz_treasury_first_boundary():
    readme = Path("README.md").read_text()

    assert "standing on-chain reserve" in readme
    assert "treasury mode first" in readme
    assert "balance cycle" in readme
