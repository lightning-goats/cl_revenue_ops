import pytest
import time
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
    ]


def test_internal_knobs_are_not_public():
    cfg = Config()

    assert "enable_vegas_reflex" not in cfg.public_runtime_keys()
    assert "thompson_prior_std_fee" not in cfg.public_runtime_keys()
    assert "sling_target_sink" not in cfg.public_runtime_keys()


def test_auto_band_config_defaults_enabled():
    cfg = Config()

    assert cfg.auto_band_enabled is True
    assert cfg.auto_band_min_observations == 20
    assert cfg.auto_band_sigma == 2.0
    assert cfg.auto_band_min_width_ppm == 50
    assert cfg.auto_band_recalibrate_interval == 10


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


def test_revenue_fee_debug_reports_effective_auto_band_details():
    peer_id = "02" + "a" * 64
    mod = load_plugin_module()
    mod.database = MagicMock()
    mod.config = Config()
    mod.fee_controller = MagicMock()
    mod.database.get_all_fee_strategy_states.return_value = [
        {
            "channel_id": "123x456x0",
            "is_sleeping": 0,
            "sleep_until": 0,
            "last_update": int(time.time()) - 3600,
            "forward_count_since_update": 5,
            "last_broadcast_fee_ppm": 220,
            "last_revenue_rate": 1.5,
        }
    ]
    mod.database.get_all_channel_states.return_value = [
        {
            "channel_id": "123x456x0",
            "peer_id": peer_id,
            "state": "balanced",
        }
    ]
    mod.fee_controller._get_effective_dynamic_fee_autoband_ppm.return_value = (200, 300, "auto")
    mod.fee_controller._get_channel_auto_band.return_value = SimpleNamespace(
        min_ppm=200,
        max_ppm=300,
        optimal_fee_ppm=250,
        posterior_std=25.0,
        observation_count=27,
        sigma=2.0,
        min_width_ppm=50,
        last_calibrated=1_773_506_400,
        source="auto",
    )

    result = mod.revenue_fee_debug(mod.plugin)

    channel = result["channels"][0]
    assert channel["effective_autoband"] == {
        "min_ppm": 200,
        "max_ppm": 300,
        "source": "auto",
    }
    assert channel["auto_band"]["optimal_fee_ppm"] == 250
    assert channel["auto_band"]["observation_count"] == 27


@pytest.mark.parametrize(
    ("action", "peer_id", "kwargs"),
    [
        ("set", "02" + "a" * 64, {"strategy": "static", "fee_ppm": 500}),
        ("delete", "02" + "a" * 64, {}),
        ("tag", "02" + "a" * 64, {"tag": "whale"}),
        ("untag", "02" + "a" * 64, {"tag": "whale"}),
        ("batch", None, {"updates": [{"peer_id": "02" + "a" * 64, "strategy": "hive"}]}),
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
    policy.to_dict.return_value = {"peer_id": "02" + "a" * 64, "strategy": "hive"}
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
