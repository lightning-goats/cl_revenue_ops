from unittest.mock import MagicMock

import pytest

from modules.config import Config
from tests.plugin_test_utils import load_plugin_module
from tests.test_operator_surface import _run_init_with_stubbed_dependencies


CAPTURE_OPTION = "revenue-ops-fee-replay-capture-enabled"


def test_capture_config_defaults_off_and_is_snapshotted():
    cfg = Config()

    assert cfg.fee_replay_capture_enabled is False
    assert cfg.snapshot().fee_replay_capture_enabled is False


def test_capture_plugin_option_is_dynamic_and_default_off():
    mod = load_plugin_module()

    option = mod.plugin.options[CAPTURE_OPTION]
    assert option["default"] == "false"
    assert option["dynamic"] is True
    assert option["on_change"] is mod._on_fee_replay_capture_change


def test_capture_option_is_parsed_during_init(monkeypatch):
    mod = load_plugin_module()

    cfg = _run_init_with_stubbed_dependencies(
        mod,
        monkeypatch,
        {CAPTURE_OPTION: "true"},
    )

    assert cfg.fee_replay_capture_enabled is True


def test_capture_callback_updates_config_and_manager():
    mod = load_plugin_module()
    manager = MagicMock()
    mod.config = Config()
    mod.fee_controller = MagicMock(_fee_capture=manager)

    mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.fee_replay_capture_enabled is True
    manager.set_enabled.assert_called_once_with(True, timeout_seconds=5.0)


@pytest.mark.parametrize("invalid", ["", "maybe", 2])
def test_capture_callback_rejects_invalid_values_without_mutation(invalid):
    mod = load_plugin_module()
    manager = MagicMock()
    mod.config = Config(fee_replay_capture_enabled=True)
    mod.fee_controller = MagicMock(_fee_capture=manager)

    with pytest.raises(ValueError, match=f"{CAPTURE_OPTION} must be a boolean"):
        mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, invalid)

    assert mod.config.fee_replay_capture_enabled is True
    manager.set_enabled.assert_not_called()
