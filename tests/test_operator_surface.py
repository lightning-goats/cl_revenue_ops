from unittest.mock import MagicMock

from modules.config import Config
from tests.test_plugin_audit_regressions import _load_plugin_module


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
    mod = _load_plugin_module()
    mod.database = MagicMock()
    mod.config = Config(
        paused=True,
        daily_budget_sats=1200,
        min_fee_ppm=15,
        max_fee_ppm=2500,
    )
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
