from types import SimpleNamespace
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
    manager.set_enabled.return_value = True
    mod.config = Config()
    mod.fee_controller = MagicMock(_fee_capture=manager)

    mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.fee_replay_capture_enabled is True
    manager.set_enabled.assert_called_once_with(True, timeout_seconds=5.0)


def test_capture_callback_rejects_enable_when_manager_does_not_start():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config()
    mod.fee_controller = MagicMock(_fee_capture=manager)
    mod.plugin.log.reset_mock()

    with pytest.raises(ValueError, match="could not be enabled"):
        mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.fee_replay_capture_enabled is False
    manager.set_enabled.assert_called_once_with(True, timeout_seconds=5.0)
    mod.plugin.log.assert_not_called()


def test_capture_callback_records_disable_while_manager_is_draining():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config(fee_replay_capture_enabled=True)
    mod.fee_controller = MagicMock(_fee_capture=manager)
    mod.plugin.log.reset_mock()

    mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, False)

    assert mod.config.fee_replay_capture_enabled is False
    manager.set_enabled.assert_called_once_with(False, timeout_seconds=5.0)
    mod.plugin.log.assert_called_once_with(
        "FEE REPLAY CAPTURE: disabled; writer is still draining",
        level="warn",
    )


@pytest.mark.parametrize(
    ("new_value", "initial_value"),
    [(True, False), (False, True)],
)
def test_capture_callback_manager_exception_precedes_config_and_success_log(
    new_value,
    initial_value,
):
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.side_effect = RuntimeError("manager failed")
    mod.config = Config(fee_replay_capture_enabled=initial_value)
    mod.fee_controller = MagicMock(_fee_capture=manager)
    mod.plugin.log.reset_mock()

    with pytest.raises(RuntimeError, match="manager failed"):
        mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, new_value)

    assert mod.config.fee_replay_capture_enabled is initial_value
    manager.set_enabled.assert_called_once_with(new_value, timeout_seconds=5.0)
    mod.plugin.log.assert_not_called()


@pytest.mark.parametrize("controller", [None, SimpleNamespace()])
def test_capture_callback_without_manager_preserves_config_only_behavior(controller):
    mod = load_plugin_module()
    mod.config = Config()
    mod.fee_controller = controller
    mod.plugin.log.reset_mock()

    mod._on_fee_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.fee_replay_capture_enabled is True
    mod.plugin.log.assert_called_once_with(
        "FEE REPLAY CAPTURE: enabled",
        level="info",
    )


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
