from unittest.mock import MagicMock

import pytest

from modules.config import Config
from tests.plugin_test_utils import load_plugin_module
from tests.test_operator_surface import _run_init_with_stubbed_dependencies


CAPTURE_OPTION = "revenue-ops-rebalance-replay-capture-enabled"


def test_rebalance_capture_defaults_off_and_is_snapshotted():
    cfg = Config()

    assert cfg.rebalance_replay_capture_enabled is False
    assert cfg.snapshot().rebalance_replay_capture_enabled is False


def test_rebalance_capture_option_is_dynamic_and_default_off():
    mod = load_plugin_module()

    option = mod.plugin.options[CAPTURE_OPTION]
    assert option["default"] == "false"
    assert option["dynamic"] is True
    assert option["on_change"] is mod._on_rebalance_replay_capture_change


def test_rebalance_capture_callback_rolls_back_when_manager_cannot_enable():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config()
    mod.rebalancer = MagicMock(rebalance_engine_v2=MagicMock(rebalance_capture_manager=manager))

    with pytest.raises(ValueError, match="could not be enabled"):
        mod._on_rebalance_replay_capture_change(mod.plugin, CAPTURE_OPTION, True)

    assert mod.config.rebalance_replay_capture_enabled is False
    manager.set_enabled.assert_called_once_with(True, timeout_seconds=5.0)


def test_rebalance_capture_option_is_parsed_at_init(monkeypatch):
    mod = load_plugin_module()

    cfg = _run_init_with_stubbed_dependencies(mod, monkeypatch, {CAPTURE_OPTION: "true"})

    assert cfg.rebalance_replay_capture_enabled is True



def test_rebalance_capture_disable_failure_rolls_back_config():
    mod = load_plugin_module()
    manager = MagicMock()
    manager.set_enabled.return_value = False
    mod.config = Config(rebalance_replay_capture_enabled=True)
    mod.rebalancer = MagicMock(rebalance_engine_v2=MagicMock(rebalance_capture_manager=manager))

    with pytest.raises(ValueError, match="could not be disabled"):
        mod._on_rebalance_replay_capture_change(mod.plugin, CAPTURE_OPTION, False)

    assert mod.config.rebalance_replay_capture_enabled is True
    manager.set_enabled.assert_called_once_with(False, timeout_seconds=5.0)
