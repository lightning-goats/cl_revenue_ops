import pytest
from unittest.mock import MagicMock
from pathlib import Path
from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


def test_public_runtime_keys_are_safety_only():
    cfg = Config()

    assert cfg.public_runtime_keys() == [
        "paused",
        "daily_budget_sats",
        "min_fee_ppm",
        "max_fee_ppm",
    ]


def test_internal_knobs_are_not_public():
    cfg = Config()

    assert "enable_vegas_reflex" not in cfg.public_runtime_keys()
    assert "thompson_prior_std_fee" not in cfg.public_runtime_keys()
    assert "sling_target_sink" not in cfg.public_runtime_keys()


def test_public_runtime_dict_returns_only_public_keys():
    cfg = Config(paused=True, daily_budget_sats=1200, min_fee_ppm=15, max_fee_ppm=2500)

    assert cfg.public_runtime_dict() == {
        "paused": True,
        "daily_budget_sats": 1200,
        "min_fee_ppm": 15,
        "max_fee_ppm": 2500,
    }


def test_runtime_key_classification_distinguishes_public_deprecated_and_internal():
    cfg = Config()

    assert cfg.classify_runtime_key("paused") == "public"
    assert cfg.classify_runtime_key("enable_vegas_reflex") == "internal"
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


def _default_plugin_options(mod):
    return {
        name: registration["default"]
        for name, registration in mod.plugin.options.items()
        if "default" in registration
    }


def _run_init_with_stubbed_dependencies(mod, monkeypatch, option_overrides=None):
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
    fake_db.get_all_config_overrides.return_value = {}
    fake_db.get_config_version.return_value = 0

    fake_rpc = MagicMock()
    fake_rpc.plugin.return_value = {"plugins": []}
    fake_rpc.listplugins.return_value = {"plugins": []}
    fake_rpc.listforwards.return_value = {"forwards": []}
    fake_rpc.listpeers.return_value = {"peers": []}

    fake_proxy = MagicMock()
    fake_proxy.rpc = fake_rpc
    fake_proxy._executor = MagicMock()
    fake_proxy._async_executor = MagicMock()

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

    assert "revenue-ops-planner-execute-closes" in mod.plugin.options
    assert mod.plugin.options["revenue-ops-planner-execute-closes"]["default"] == "false"


def test_planner_cycle_limit_defaults_match_config():
    mod = load_plugin_module()
    cfg = Config()

    assert cfg.planner_max_opens_per_cycle == 1
    assert cfg.planner_max_closes_per_cycle == 0
    assert mod.plugin.options["revenue-ops-planner-max-opens-per-cycle"]["default"] == "1"
    assert mod.plugin.options["revenue-ops-planner-max-closes-per-cycle"]["default"] == "0"


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
            "revenue-ops-planner-max-closes-per-cycle": "0",
        },
    )

    assert cfg.planner_max_opens_per_cycle == 3
    assert cfg.planner_max_closes_per_cycle == 0


def test_revenue_config_list_mutable_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "list-mutable")

    assert result["mutable_keys"] == [
        "daily_budget_sats",
        "max_fee_ppm",
        "min_fee_ppm",
        "paused",
    ]
    assert result["count"] == 4


def test_revenue_config_get_without_key_returns_public_controls_only():
    mod = _load_operator_surface_module()

    result = mod.revenue_config(mod.plugin, "get")

    assert result["config"] == {
        "paused": True,
        "daily_budget_sats": 1200,
        "min_fee_ppm": 15,
        "max_fee_ppm": 2500,
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
    }
    assert "config" not in result


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


def test_readme_examples_no_longer_advertise_internal_knob_tuning():
    readme = Path("README.md").read_text()

    assert "paused" in readme
    assert "daily_budget_sats" in readme
    assert "min_fee_ppm" in readme
    assert "max_fee_ppm" in readme
    assert "decision explainability" in readme
    assert "revenue-config set enable_vegas_reflex false" not in readme
    assert "lightning-cli revenue-policy set" not in readme
