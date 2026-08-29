import pytest
from unittest.mock import MagicMock, call
from pathlib import Path
from types import SimpleNamespace
from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


FEE_AUTHORITY_OPTION = "revenue-ops-fee-authority-enabled"


def test_fee_authority_plugin_option_is_dynamic_boolean_and_default_on():
    mod = load_plugin_module()

    option = mod.plugin.options[FEE_AUTHORITY_OPTION]
    assert option["default"] is True
    assert option["opt_type"] == "bool"
    assert option["dynamic"] is True
    assert option["on_change"] is mod._on_fee_authority_change


def test_public_runtime_keys_are_safety_only():
    cfg = Config()

    assert cfg.public_runtime_keys() == [
        "paused",
        "daily_budget_sats",
        # E-3 (2026-07 econ audit): weekly cap runtime control
        "weekly_budget_sats",
        "growth_budget_enabled",
        "growth_budget_earned_fraction",
        "growth_budget_experiment_fraction",
        "growth_budget_max_extra_sats",
        "growth_budget_hard_ceiling_sats",
        "min_fee_ppm",
        # E-2 (2026-07 econ audit): class-aware saturated/source floor
        "min_fee_ppm_saturated",
        "acquisition_experiment_enabled",
        "max_fee_ppm",
        "fee_profile",
        "fee_market_boundary_enabled",
        "fee_market_boundary_min_competitors",
        "fee_market_boundary_margin_ppm",
        "fee_market_boundary_margin_ratio",
        "fee_market_boundary_max_downshift_ratio",
        "fee_market_boundary_cache_seconds",
        "capex_probability_budget_bonus",
        "receivable_ratio_target",
        "receivable_ratio_floor",
        "drain_fee_discount_max",
        "node_drain_bias_enabled",
        "node_drain_bias_max",
        # Dynamic htlc_max flow valve (H-2, 2026-07-03 audit)
        "enable_dynamic_htlcmax",
        "htlcmax_source_pct",
        "htlcmax_sink_pct",
        "htlcmax_balanced_pct",
        "econ_shadow_enabled",
        "econ_governor_rebalance_enabled",
        "econ_governor_fees_enabled",
        "econ_arbiter_enabled",
        "econ_cycle_rebalance_enabled",
        "econ_ev_populated",
        "econ_conflict_rules_extended",
        "authority_level",
        "risk_profile",
        # LN+ liquidity swap automation
    ]


def test_internal_knobs_are_not_public():
    cfg = Config()

    assert "fee_authority_enabled" not in cfg.public_runtime_keys()
    assert "enable_vegas_reflex" not in cfg.public_runtime_keys()
    assert "thompson_prior_std_fee" not in cfg.public_runtime_keys()
    assert "askrene_layers" not in cfg.public_runtime_keys()


def test_public_runtime_dict_returns_only_public_keys():
    cfg = Config(paused=True, daily_budget_sats=1200, min_fee_ppm=15, max_fee_ppm=2500)

    assert cfg.public_runtime_dict() == {
        "paused": True,
        "daily_budget_sats": 1200,
        "weekly_budget_sats": 35000,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.10,
        "growth_budget_max_extra_sats": 2000,
        "growth_budget_hard_ceiling_sats": 10000,
        "min_fee_ppm": 15,
        "min_fee_ppm_saturated": 0,
        "acquisition_experiment_enabled": True,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "capex_probability_budget_bonus": 0.0,
        "receivable_ratio_target": 0.3,
        "receivable_ratio_floor": 0.2,
        "drain_fee_discount_max": 0.0,
        "node_drain_bias_enabled": False,
        "node_drain_bias_max": 0.3,
        "enable_dynamic_htlcmax": True,
        "htlcmax_source_pct": 0.85,
        "htlcmax_sink_pct": 0.85,
        "htlcmax_balanced_pct": 0.85,
        "econ_shadow_enabled": True,
        "econ_governor_rebalance_enabled": True,
        "econ_governor_fees_enabled": True,
        "econ_arbiter_enabled": True,
        "econ_cycle_rebalance_enabled": True,
        "econ_ev_populated": True,
        "econ_conflict_rules_extended": True,
        "authority_level": "capital",
        "risk_profile": "custom",
    }


def test_runtime_key_classification_distinguishes_public_deprecated_and_internal():
    cfg = Config()

    assert cfg.classify_runtime_key("paused") == "public"
    assert cfg.classify_runtime_key("enable_vegas_reflex") == "internal"
    assert cfg.classify_runtime_key("askrene_layers") == "internal"
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
    fake_db.forward_archive = MagicMock()
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
    mod._test_fake_db = fake_db
    mod._test_fake_proxy = fake_proxy
    mod._test_threads = []

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            mod._test_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr(mod, "ThreadSafePluginProxy", lambda plugin: fake_proxy)
    monkeypatch.setattr(mod, "Database", lambda *args, **kwargs: fake_db)
    monkeypatch.setattr(mod, "PolicyManager", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "ChannelProfitabilityAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "FlowAnalyzer", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "FeeController", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(mod, "EVRebalancer", lambda *args, **kwargs: MagicMock(
        set_profitability_analyzer=MagicMock(),
    ))
    monkeypatch.setattr(mod.threading, "Thread", DummyThread)

    mod.init(options, {}, mod.plugin)
    return mod.config


def test_fee_authority_option_is_parsed_during_init_and_initializes_gate(monkeypatch):
    mod = load_plugin_module()

    cfg = _run_init_with_stubbed_dependencies(
        mod,
        monkeypatch,
        {FEE_AUTHORITY_OPTION: "false"},
    )

    assert cfg.fee_authority_enabled is False
    status = mod.fee_authority_gate.snapshot()
    assert status.enabled is False
    assert status.generation == 1
    assert status.reason == "init"


def test_init_logs_native_route_rebalancing(monkeypatch):
    mod = load_plugin_module()
    _run_init_with_stubbed_dependencies(mod, monkeypatch)

    messages = [call.args[0] for call in mod.plugin.log.call_args_list if call.args]

    assert any("native route execution" in message.lower() for message in messages)
    assert not any("getroute + sendpay" in message for message in messages)


def test_init_starts_fail_isolated_forward_archive_daemon(monkeypatch):
    mod = load_plugin_module()
    fake_sync = MagicMock()
    factory = MagicMock(return_value=fake_sync)
    monkeypatch.setattr(
        mod,
        "ForwardArchiveSynchronizer",
        factory,
        raising=False,
    )

    _run_init_with_stubbed_dependencies(mod, monkeypatch)

    factory.assert_called_once_with(
        mod._test_fake_rpc,
        mod._test_fake_db.forward_archive,
        mod._test_fake_proxy.log,
    )
    assert mod.forward_archive_sync is fake_sync
    archive_thread = next(
        thread
        for thread in mod._test_threads
        if thread.kwargs.get("name") == "forward-archive"
    )
    assert archive_thread.kwargs["daemon"] is True
    assert callable(archive_thread.kwargs["target"])
    fake_sync.sync_once.assert_not_called()


def test_forward_archive_daemon_isolates_cycle_failure(monkeypatch):
    mod = load_plugin_module()
    fake_sync = MagicMock()
    fake_sync.sync_once.side_effect = RuntimeError("temporary archive failure")
    monkeypatch.setattr(
        mod,
        "ForwardArchiveSynchronizer",
        MagicMock(return_value=fake_sync),
        raising=False,
    )
    fake_shutdown = MagicMock()
    fake_shutdown.is_set.return_value = False
    fake_shutdown.wait.side_effect = [False, True]
    monkeypatch.setattr(mod, "shutdown_event", fake_shutdown)

    _run_init_with_stubbed_dependencies(mod, monkeypatch)
    heartbeat = MagicMock()
    monkeypatch.setattr(mod, "_record_loop_heartbeat", heartbeat)
    archive_thread = next(
        thread
        for thread in mod._test_threads
        if thread.kwargs.get("name") == "forward-archive"
    )

    archive_thread.kwargs["target"]()

    fake_sync.sync_once.assert_called_once_with()
    heartbeat.assert_called_once_with(
        "forward-archive",
        interval_seconds=900,
    )
    assert fake_shutdown.wait.call_args_list[-2:] == [call(60), call(900)]
    messages = [
        logged.args[0]
        for logged in mod.plugin.log.call_args_list
        if logged.args
    ]
    assert any(
        "Error in forward archive sync: temporary archive failure" in message
        for message in messages
    )


def test_forward_archive_daemon_logs_checkpointed_backlog_without_error(
    monkeypatch,
):
    mod = load_plugin_module()
    fake_sync = MagicMock()
    fake_sync.sync_once.return_value = SimpleNamespace(
        caught_up=False,
        backlog_family="updated",
        created_pages=0,
        updated_pages=200,
    )
    monkeypatch.setattr(
        mod,
        "ForwardArchiveSynchronizer",
        MagicMock(return_value=fake_sync),
        raising=False,
    )
    fake_shutdown = MagicMock()
    fake_shutdown.is_set.return_value = False
    fake_shutdown.wait.side_effect = [False, True]
    monkeypatch.setattr(mod, "shutdown_event", fake_shutdown)
    _run_init_with_stubbed_dependencies(mod, monkeypatch)
    archive_thread = next(
        thread for thread in mod._test_threads
        if thread.kwargs.get("name") == "forward-archive"
    )

    archive_thread.kwargs["target"]()

    backlog_logs = [
        call for call in mod.plugin.log.call_args_list
        if call.args and "backlog checkpointed" in call.args[0]
    ]
    assert len(backlog_logs) == 1
    assert backlog_logs[0].kwargs.get("level") == "info"
    assert "family=updated" in backlog_logs[0].args[0]
    assert "created_pages=0" in backlog_logs[0].args[0]
    assert "updated_pages=200" in backlog_logs[0].args[0]
    assert not any(
        call.kwargs.get("level") == "error"
        for call in mod.plugin.log.call_args_list
        if call.args and "page limit" in call.args[0]
    )


def test_init_keeps_plugin_available_when_forward_archive_sync_is_unavailable(
    monkeypatch,
):
    mod = load_plugin_module()
    monkeypatch.setattr(
        mod,
        "ForwardArchiveSynchronizer",
        MagicMock(side_effect=RuntimeError("archive schema unavailable")),
        raising=False,
    )

    _run_init_with_stubbed_dependencies(mod, monkeypatch)

    assert mod.forward_archive_sync is None
    messages = [
        call.args[0]
        for call in mod.plugin.log.call_args_list
        if call.args
    ]
    assert any(
        "Forward archive synchronization unavailable" in message
        and "archive schema unavailable" in message
        for message in messages
    )


def test_revenue_forward_history_is_bounded_read_only_delegate():
    mod = _load_operator_surface_module()
    expected = {"coverage": [], "rows": []}
    mod.database.forward_archive.history.return_value = expected

    result = mod.revenue_forward_history(
        mod.plugin,
        history_since=1786492800,
        history_until=1786579200,
        channel_id="1:2:3",
        limit=50,
    )

    mod.database.forward_archive.history.assert_called_once_with(
        1786492800,
        1786579200,
        "1x2x3",
        50,
    )
    assert result == expected
    calls = str(mod.database.method_calls).lower()
    assert not any(
        name in calls
        for name in (
            "sync",
            "apply",
            "repair",
            "fee",
            "rebalance",
            "reserve",
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "history_since": 1786492801,
            "history_until": 1786579200,
        },
        {
            "history_since": 1786492800,
            "history_until": 1786492800,
        },
        {
            "history_since": 1786492800,
            "history_until": 1786579200,
            "limit": 5001,
        },
        {
            "history_since": 1786492800,
            "history_until": 1786492800 + 401 * 86400,
        },
        {
            "history_since": 1786492800,
            "history_until": 1786579200,
            "limit": True,
        },
        {
            "history_since": 1786492800,
            "history_until": 1786579200,
            "channel_id": 123,
        },
    ],
)
def test_revenue_forward_history_rejects_unbounded_or_unaligned_requests(
    kwargs,
):
    mod = _load_operator_surface_module()

    result = mod.revenue_forward_history(mod.plugin, **kwargs)

    assert "error" in result
    mod.database.forward_archive.history.assert_not_called()


def test_revenue_forward_history_is_neutral_when_archive_is_absent():
    mod = _load_operator_surface_module()
    mod.database = SimpleNamespace()

    result = mod.revenue_forward_history(
        mod.plugin,
        history_since=1786492800,
        history_until=1786579200,
    )

    assert result == {"error": "Forward archive not initialized"}


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
    result = mod.revenue_rebalance_debug(mod.plugin)

    assert result["last_decision"]["reason"] == "no_rebalance_candidates"
    assert result["last_cycle"]["summary"]["considered_pairs"] == 1
    assert (
        result["last_cycle"]["considered_candidates"][0]["score_decomposition"]["rejection_reason"]
        == "route_over_budget"
    )


def test_rebalance_tuning_plugin_options_are_dynamic():
    mod = load_plugin_module()

    for option in (
        "revenue-ops-rebalance-hold-margin",
        "revenue-ops-pair-fee-cap-ppm",
    ):
        assert mod.plugin.options[option]["dynamic"] is True
        assert mod.plugin.options[option]["on_change"] is mod._on_rebalance_tuning_change


def test_rebalance_tuning_setconfig_updates_live_config():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(
        rebalance_hold_margin=0.0,
        pair_fee_cap_ppm=1000,
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

    assert mod.config.rebalance_hold_margin == 0.25
    assert mod.config.pair_fee_cap_ppm == 750


def test_rebalance_tuning_setconfig_rejects_out_of_range_values():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(rebalance_hold_margin=0.0)

    # final_score is sats-denominated; the hold margin admits up to 1000 sats.
    with pytest.raises(ValueError, match="<= 1000.0"):
        mod._on_rebalance_tuning_change(
            mod.plugin,
            "revenue-ops-rebalance-hold-margin",
            "1500",
        )


def test_askrene_layer_option_defaults_to_standalone():
    mod = load_plugin_module()

    assert mod.plugin.options["revenue-ops-askrene-layers"]["default"] == "standalone"
    assert "askrene getroutes" in mod.plugin.options["revenue-ops-rebalance-router"]["description"]
    assert "standalone" in mod.plugin.options["revenue-ops-askrene-layers"]["description"]


def test_revenue_config_list_mutable_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "list-mutable")

    assert result["mutable_keys"] == [
        "acquisition_experiment_enabled",
        "authority_level",
        "capex_probability_budget_bonus",
        "daily_budget_sats",
        "drain_fee_discount_max",
        "econ_arbiter_enabled",
        "econ_conflict_rules_extended",
        "econ_cycle_rebalance_enabled",
        "econ_ev_populated",
        "econ_governor_fees_enabled",
        "econ_governor_rebalance_enabled",
        "econ_shadow_enabled",
        "enable_dynamic_htlcmax",
        "fee_market_boundary_cache_seconds",
        "fee_market_boundary_enabled",
        "fee_market_boundary_margin_ppm",
        "fee_market_boundary_margin_ratio",
        "fee_market_boundary_max_downshift_ratio",
        "fee_market_boundary_min_competitors",
        "fee_profile",
        "growth_budget_earned_fraction",
        "growth_budget_enabled",
        "growth_budget_experiment_fraction",
        "growth_budget_hard_ceiling_sats",
        "growth_budget_max_extra_sats",
        "htlcmax_balanced_pct",
        "htlcmax_sink_pct",
        "htlcmax_source_pct",
        "max_fee_ppm",
        "min_fee_ppm",
        "min_fee_ppm_saturated",
        "node_drain_bias_enabled",
        "node_drain_bias_max",
        "paused",
        "receivable_ratio_floor",
        "receivable_ratio_target",
        "risk_profile",
        "weekly_budget_sats",
    ]
    assert result["count"] == 38


def test_revenue_config_get_without_key_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "get")

    assert result["config"] == {
        "paused": True,
        "daily_budget_sats": 1200,
        "weekly_budget_sats": 35000,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.10,
        "growth_budget_max_extra_sats": 2000,
        "growth_budget_hard_ceiling_sats": 10000,
        "min_fee_ppm": 15,
        "min_fee_ppm_saturated": 0,
        "acquisition_experiment_enabled": True,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "capex_probability_budget_bonus": 0.0,
        "receivable_ratio_target": 0.3,
        "receivable_ratio_floor": 0.2,
        "drain_fee_discount_max": 0.0,
        "node_drain_bias_enabled": False,
        "node_drain_bias_max": 0.3,
        "enable_dynamic_htlcmax": True,
        "htlcmax_source_pct": 0.85,
        "htlcmax_sink_pct": 0.85,
        "htlcmax_balanced_pct": 0.85,
        "econ_shadow_enabled": True,
        "econ_governor_rebalance_enabled": True,
        "econ_governor_fees_enabled": True,
        "econ_arbiter_enabled": True,
        "econ_cycle_rebalance_enabled": True,
        "econ_ev_populated": True,
        "econ_conflict_rules_extended": True,
        "authority_level": "capital",
        "risk_profile": "custom",
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
    mod._invalidate_total_cost_budget_memo = MagicMock()
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
    mod._invalidate_total_cost_budget_memo.assert_called_once_with()


def test_revenue_config_failed_update_keeps_budget_memo():
    mod = _load_operator_surface_module()
    mod._invalidate_total_cost_budget_memo = MagicMock()
    mod.config.update_runtime = MagicMock(return_value={"status": "error"})

    result = mod.revenue_config(mod.plugin, "set", "daily_budget_sats", "1000")

    assert result["status"] == "error"
    mod._invalidate_total_cost_budget_memo.assert_not_called()


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


def test_revenue_report_summary_includes_financials_and_live_channels():
    mod = _load_report_surface_module()
    mod.policy_manager.get_all_policies.return_value = []
    mod.profitability_analyzer.get_tlv.return_value = {"tlv_sats": 8_000_000}
    mod.profitability_analyzer.get_pnl_summary.return_value = {
        "net_profit_sats": 44,
        "operating_margin_pct": 86.27,
        "gross_revenue_sats": 51,
        "opex_sats": 7,
        "rebalance_cost_sats": 7,
        "closure_cost_sats": 0,
        "volume_sats": 1_140_000,
        "forward_count": 42,
    }
    mod.profitability_analyzer.calculate_roc.return_value = {
        "annualized_roc_pct": 0.2,
    }
    mod.profitability_analyzer.identify_bleeders.return_value = []
    mod.database.get_all_channel_states.return_value = [
        {"channel_id": "100x1x0", "state": "source"},
        {"short_channel_id": "200x1x0", "state": "sink"},
    ]

    result = mod.revenue_report(mod.plugin, "summary")

    assert result["financial_health"]["net_profit_sats"] == 44
    assert result["period"]["forward_count"] == 42
    assert result["channels"] == {
        "total": 2,
        "by_state": {"source": 1, "sink": 1},
    }
    assert result["policies"]["total"] == 0
    assert result["warnings"] == []


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
        "spent_24h_sats": 77,
        "reserved_24h_sats": 27,
        "spent_by_category": {
            "channel_open": 50,
            "channel_close": 15,
            "boltz": 5,
            "misc_ops": 7,
        },
        "reserved_by_category": {
            "channel_open": 20,
            "boltz": 4,
            "misc_ops": 3,
        },
        "event_count_by_category": {
            "channel_open": 1,
            "channel_close": 1,
            "misc_ops": 1,
        },
        "active_reservation_count_by_category": {
            "channel_open": 1,
            "misc_ops": 1,
        },
    }
    mod.database.get_total_routing_revenue.return_value = 500
    mod.database.get_opening_costs_since.return_value = 100
    mod.database.get_closure_costs_since.return_value = 30
    mod.database.get_cost_evidence_coverage.return_value = {
        "covered_hours": 24,
        "coverage_status": "complete",
    }
    mod._rebalance_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 11, "reserved_24h_sats": 2}
    )

    result = mod._total_cost_budget_status(window_hours=24)

    assert isinstance(result["timestamp"], int)
    assert result["generated_at"] == result["timestamp"]
    assert result["ttl_seconds"] == 1800
    assert result["coverage_hours"] == 24
    assert result["covered_hours"] == 24
    assert result["coverage_status"] == "complete"
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
        "boltz": 5,
    }
    assert result["open_close_cost_visibility"] == {
        "canonical_open_cost_available": True,
        "canonical_close_cost_available": True,
        "pending_open_close_spend_events": 1,
        "excluded_from_generic_totals_to_avoid_double_count": True,
        "excluded_open_close_spend_sats": 65,
        "reserved_open_close_sats": 20,
    }


def _total_cost_budget_module_with_stub_components():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(
        reservation_timeout_hours=4,
        daily_budget_sats=2000,
    )
    mod.database = MagicMock()
    mod.database.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 0,
        "reserved_24h_sats": 0,
        "spent_by_category": {},
        "reserved_by_category": {},
        "event_count_by_category": {},
        "active_reservation_count_by_category": {},
    }
    mod.database.get_total_routing_revenue.return_value = 0
    mod.database.get_opening_costs_since.return_value = 0
    mod.database.get_closure_costs_since.return_value = 0
    mod._rebalance_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    return mod


def test_total_cost_budget_coverage_unknown_when_unmeasurable():
    """No measurement basis (coverage lookup fails) must surface an honest
    unknown — never a fabricated 'complete' echo of the requested window."""
    mod = _total_cost_budget_module_with_stub_components()
    mod.database.get_cost_evidence_coverage.side_effect = RuntimeError("db gone")

    result = mod._total_cost_budget_status(window_hours=24)

    assert result["covered_hours"] is None
    assert result["coverage_hours"] is None
    assert result["coverage_status"] == "unknown"
    assert result["window_hours"] == 24


def test_total_cost_budget_coverage_unknown_on_garbage_payload():
    mod = _total_cost_budget_module_with_stub_components()
    mod.database.get_cost_evidence_coverage.return_value = {
        "covered_hours": "not-a-number",
        "coverage_status": "complete",
    }

    result = mod._total_cost_budget_status(window_hours=24)

    assert result["covered_hours"] is None
    assert result["coverage_status"] == "unknown"


def test_total_cost_budget_reports_measured_partial_coverage():
    """Coverage fields must reflect the measured evidence span, not the
    requested window (external consumers read covered_hours numerically)."""
    mod = _total_cost_budget_module_with_stub_components()
    mod.database.get_cost_evidence_coverage.return_value = {
        "covered_hours": 3.0,
        "coverage_status": "partial",
    }

    result = mod._total_cost_budget_status(window_hours=24)

    assert result["covered_hours"] == 3.0
    assert result["coverage_hours"] == 3.0
    assert result["coverage_status"] == "partial"
    mod.database.get_cost_evidence_coverage.assert_called_once_with(window_hours=24)


def test_total_cost_budget_reports_pending_open_close_visibility_delay():
    mod = load_plugin_module()
    mod.config = SimpleNamespace(
        reservation_timeout_hours=4,
        daily_budget_sats=2000,
    )
    mod.database = MagicMock()
    mod.database.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 50,
        "reserved_24h_sats": 0,
        "spent_by_category": {"channel_close": 50},
        "reserved_by_category": {},
        "event_count_by_category": {"channel_close": 1},
    }
    mod.database.get_total_routing_revenue.return_value = 0
    mod.database.get_opening_costs_since.return_value = 0
    mod.database.get_closure_costs_since.return_value = 0
    mod._rebalance_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )

    result = mod._total_cost_budget_status(window_hours=24)

    assert result["actual_spent_sats"] == 0
    visibility = result["open_close_cost_visibility"]
    assert visibility["canonical_close_cost_available"] is False
    assert visibility["pending_open_close_spend_events"] == 1
    assert visibility["excluded_open_close_spend_sats"] == 50
    assert visibility["excluded_from_generic_totals_to_avoid_double_count"] is True


def test_revenue_status_operator_controls_hide_internal_knob_dump():
    mod = _load_operator_surface_module()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []

    result = mod.revenue_status(mod.plugin)

    assert result["operator_controls"]["values"] == {
        "paused": True,
        "daily_budget_sats": 1200,
        "weekly_budget_sats": 35000,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.10,
        "growth_budget_max_extra_sats": 2000,
        "growth_budget_hard_ceiling_sats": 10000,
        "min_fee_ppm": 15,
        "min_fee_ppm_saturated": 0,
        "acquisition_experiment_enabled": True,
        "max_fee_ppm": 2500,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "capex_probability_budget_bonus": 0.0,
        "receivable_ratio_target": 0.3,
        "receivable_ratio_floor": 0.2,
        "drain_fee_discount_max": 0.0,
        "node_drain_bias_enabled": False,
        "node_drain_bias_max": 0.3,
        "enable_dynamic_htlcmax": True,
        "htlcmax_source_pct": 0.85,
        "htlcmax_sink_pct": 0.85,
        "htlcmax_balanced_pct": 0.85,
        "econ_shadow_enabled": True,
        "econ_governor_rebalance_enabled": True,
        "econ_governor_fees_enabled": True,
        "econ_arbiter_enabled": True,
        "econ_cycle_rebalance_enabled": True,
        "econ_ev_populated": True,
        "econ_conflict_rules_extended": True,
        "authority_level": "capital",
        "risk_profile": "custom",
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


def test_readme_describes_standalone_surfaces():
    readme = Path("README.md").read_text()

    assert "paused" in readme
    assert "daily_budget_sats" in readme
    assert "min_fee_ppm" in readme
    assert "max_fee_ppm" in readme
    assert "decision explainability" in readme
    assert "revenue-config set enable_vegas_reflex false" not in readme
    assert "lightning-cli revenue-policy set" not in readme
    assert "fully standalone" in readme
    assert "revenue-hive-hints-status" not in readme


def test_agent_docs_describe_standalone_invariants():
    agent_docs = Path("AGENTS.md").read_text()

    assert "standalone" in agent_docs
    assert "fully independent" in agent_docs


def test_revenue_profitability_defaults_to_cached_analysis():
    mod = load_plugin_module()
    mod.profitability_analyzer = MagicMock()
    mod.profitability_analyzer.analyze_all_channels.return_value = {}

    result = mod.revenue_profitability(mod.plugin)

    assert "error" not in result
    # Default must NOT force a full re-analysis on the dispatch thread.
    mod.profitability_analyzer.analyze_all_channels.assert_called_once_with(force=False)


def test_revenue_profitability_refresh_forces_reanalysis():
    mod = load_plugin_module()
    mod.profitability_analyzer = MagicMock()
    mod.profitability_analyzer.analyze_all_channels.return_value = {}

    result = mod.revenue_profitability(mod.plugin, refresh=True)

    assert "error" not in result
    mod.profitability_analyzer.analyze_all_channels.assert_called_once_with(force=True)
